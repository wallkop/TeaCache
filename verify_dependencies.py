try:
    import diffusers
    import transformers
    import protobuf
    import tokenizers
    import sentencepiece
    import imageio
    print('All dependencies installed successfully')
except ImportError as e:
    print(f'Error importing dependencies: {e}')
    exit(1)
