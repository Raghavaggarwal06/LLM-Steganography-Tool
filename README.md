# IText2Bin - Llama-Zip Compression with Modal GPU

This module provides the `IText2Bin` function that compresses text using the [llama-zip](https://github.com/AlexBuz/llama-zip) compression tool, hosted on Modal GPU containers for fast inference.

## Features

- **LLM-Powered Compression**: Uses llama-zip with Q4_K_M quantized models for high compression ratios
- **GPU Acceleration**: Runs on Modal GPU containers (A10G) for fast inference
- **Base64 Output**: Returns base64-encoded compressed data with 8-bit header
- **Automatic Model Management**: Downloads and caches the Q4_K_M model automatically

## Requirements

- **Modal account and CLI configured** (⚠️ **Required but not set up yet** - see Setup section below)
- Python 3.11+
- Modal package installed: `pip install modal`

## Configuration

The function uses the following settings:
- **Model**: Llama-3.1-8B-Instruct Q4_K_M (GGUF format)
- **Context Length**: 8192 tokens (`--n-ctx 8192`)
- **Window Overlap**: 25% (`-w 25%`)

## Usage

### Basic Usage

```python
from IText2Bin import IText2Bin

# Compress text
input_text = "Your text to compress here..."
compressed_base64 = IText2Bin(input_text)

print(f"Compressed (base64): {compressed_base64}")
```

### Using Modal Directly

```python
import modal
from IText2Bin import app, LlamaZipCompressor

with app.run():
    compressor = LlamaZipCompressor()
    result = compressor.IText2Bin.remote("Your text here...")
    print(result)
```

## Output Format

The function returns a base64-encoded string containing:
1. **8-bit header**: Single byte representing the length of compressed data in bits (0-255)
2. **Compressed binary data**: The actual compressed output from llama-zip

The format is: `base64([8-bit header][compressed binary data])`

### Decoding the Output

```python
import base64

# Decode the base64 output
decoded = base64.b64decode(compressed_base64)

# Extract header (bit length)
bit_length = decoded[0]

# Extract compressed data
compressed_data = decoded[1:]

print(f"Bit length: {bit_length}")
print(f"Compressed size: {len(compressed_data)} bytes")
```

## Limitations

- **8-bit Header Limitation**: The header can only represent bit lengths from 0-255 (31.875 bytes). For larger compressed data, the length wraps around using modulo 256. Consider using a larger header (16-bit or 32-bit) for production use with larger files.

## Setup

⚠️ **Note**: This implementation requires Modal to be set up. The Modal setup is not yet configured.

### Modal Setup (To be done later)

1. **Install Modal CLI**:
   ```bash
   pip install modal
   modal token new
   ```

2. **Deploy the Modal app**:
   ```bash
   modal deploy IText2Bin
   ```

3. **Test locally**:
   ```bash
   python IText2Bin
   ```

## Model Information

The function automatically downloads the Llama-3.1-8B-Instruct Q4_K_M model from HuggingFace on first use. The model is cached in a Modal volume for subsequent runs.

- **Model Size**: ~4.6 GB (Q4_K_M quantized)
- **Source**: HuggingFace (bartowski/Llama-3.1-8B-Instruct-GGUF)

## Performance

Compression performance depends on:
- Input text length and structure
- GPU availability (A10G recommended)
- Model context window (8192 tokens)

Typical compression ratios for natural language text: 5-15x (compared to uncompressed text).

## Error Handling

The function raises `RuntimeError` if:
- Compression fails (llama-zip error)
- Model download fails
- File I/O errors occur

## License

This implementation uses llama-zip (see [AlexBuz/llama-zip](https://github.com/AlexBuz/llama-zip) for license information).
