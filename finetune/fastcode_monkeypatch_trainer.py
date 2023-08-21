import sys
from utils.monkey_patches import replace_attn_with_flash_attn, replace_attn_with_xformer

if __name__ == '__main__':
    if any(arg in sys.argv for arg in ['--use_flash_attn', '--use-flash-attn']):
        print(f"Using flash attention")
        replace_attn_with_flash_attn()
    if any(arg in sys.argv for arg in ['--use_xformer_attn', '--use-xformer-attn']):
        print(f"Using xformer mem-efficient attention")
        replace_attn_with_xformer()

    from fastcode_trainer import main
    main()