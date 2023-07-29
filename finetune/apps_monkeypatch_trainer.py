import sys
from monkey_patches import replace_attn_with_flash_attn, replace_attn_with_xformer

if __name__ == '__main__':
    if any(arg in sys.argv for arg in ['--use_flash_attn']):
        print(f"Using flash attention")
        replace_attn_with_flash_attn()
    if any(arg in sys.argv for arg in ['--use_xformer_attn']):
        print(f"Using xformer mem-efficient attention")
        replace_attn_with_xformer()

    from apps_trainer import main
    main()