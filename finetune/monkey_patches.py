"""
Following code is borrowed, adjusted and inspired from https://raw.githubusercontent.com/oobabooga/text-generation-webui/main/modules/llama_attn_hijack.py
"""

import logging
from typing import List, Optional, Tuple

import torch
import transformers
from torch import nn
from peft.tuners.lora import LoraLayer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from einops import rearrange

import xformers.ops as xops

try:
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
except ImportError:
    logging.warning("flash_attn not installed")

def llama_xformer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
            Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None
    
    # Copied from modeling_open_llama.py
    attn_weights = None

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    attn_output = xops.memory_efficient_attention(
        query_states, key_states, value_states, attn_bias=xops.LowerTriangularMask(), p=0
    )
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)
    
    assert not use_cache, "xformer does not support use_cache=True"

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def llama_fa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )  # shape: (b, num_heads, s, head_dim)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        # reuse k, v
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # Transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states], dim=2)
    qkv = qkv.transpose(1, 3)  # shape: [b, s, 3, num_heads, head_dim]
    key_padding_mask = attention_mask

    if key_padding_mask is None:
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        max_s = q_len
        output = flash_attn_varlen_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output = output.view(bsz, q_len, -1)
    else:
        qvk = qkv.reshape(bsz, q_len, -1)
        qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        output_unpad = flash_attn_varlen_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)

    return self.o_proj(output), None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask_llama(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask

def bigcode_fa_new_attn(self, query, key, value, attention_mask=None, head_mask=None):
    ## require for lora -- need fix?
    query = query.to(torch.bfloat16)
    key = key.to(torch.bfloat16)
    value = value.to(torch.bfloat16)

    ## flash attention
    bsz, q_len, _ = query.shape
    key = key.transpose(-1,-2)[:,:,None,:]
    value = value[:,:,None,:]
    query = query.reshape(bsz, q_len, self.num_heads, self.head_dim)
    attn_output = flash_attn_func(query, key, value, dropout_p=self.attn_dropout.p, softmax_scale=None, causal=True)
    attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], self.num_heads*self.head_dim)
    return attn_output, None
    

def bigcode_xformer_new_attn(self, query, key, value, attention_mask=None, head_mask=None):    
    ## require for lora -- need fix in c_attn?
    query = query.to(torch.bfloat16)
    key = key.to(torch.bfloat16)
    value = value.to(torch.bfloat16)

    # xformer attention
    key = key.transpose(-1,-2)[:,:,None,:]
    value = value[:,:,None,:]
    query = query.reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)
    key = torch.repeat_interleave(key, self.num_heads//key.shape[2], dim=2)
    value = torch.repeat_interleave(value, self.num_heads//value.shape[2], dim=2)
    attn_output = xops.memory_efficient_attention(query, key, value, p=0, attn_bias=xops.LowerTriangularMask()) 
    attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], self.num_heads*self.head_dim)    
    return attn_output, None


def replace_attn_with_flash_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logging.warning(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
        return
    
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask_llama
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_fa_forward
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeAttention._attn = bigcode_fa_new_attn
    
def replace_attn_with_xformer():
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_xformer_forward
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeAttention._attn = bigcode_xformer_new_attn


def upcast_layer_for_flash_attention(model, torch_dtype):
    # LlamaRMSNorm layers are in fp32 after kbit_training, so we need to
    # convert them back to fp16/bf16 for flash-attn compatibility.
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.to(torch_dtype)
        if "norm" in name or "ln_" in name:
            module.to(torch_dtype)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module.to(torch_dtype)
        if "c_attn" in name or "q_attn" in name or "c_proj" in name or "c_fc" in name:
            if hasattr(module, "weight"):
                module.to(torch_dtype)

    return model