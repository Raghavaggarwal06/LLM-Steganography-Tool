from __future__ import annotations

import base64
import os
import shutil
import subprocess
from dataclasses import dataclass


class LlamaZipNotFoundError(RuntimeError):
    pass


class LlamaZipModelPathError(ValueError):
    pass


@dataclass(frozen=True)
class LlamaZipOptions:
    """
    Options matching llama-zip CLI.

    Defaults per request:
    - n_ctx: 8192
    - window_overlap: "25%"
    """

    n_ctx: int = 8192
    window_overlap: str = "25%"
    n_gpu_layers: int = -1
    compressed_format: str = "base64"


def _frame_with_1byte_len_header(payload: bytes) -> bytes:
    """
    Prepend a 1-byte header containing the payload length.

    The user asked for an "8 bit header ... includes the length of the following
    encoded bits". Since llama-zip outputs whole bytes, we store the payload
    length in BYTES (0-255), i.e. length of the following encoded data.
    """

    if len(payload) > 0xFF:
        raise ValueError(
            f"Compressed payload too large for 8-bit length header: {len(payload)} bytes"
        )
    return bytes([len(payload)]) + payload


def _run_llama_zip_compress_base64(
    *,
    text: str,
    model_path: str,
    options: LlamaZipOptions,
    llama_zip_bin: str = "llama-zip",
) -> bytes:
    """
    Returns base64 (ASCII) bytes of llama-zip compressed output.
    """

    exe = shutil.which(llama_zip_bin)
    if not exe:
        raise LlamaZipNotFoundError(
            f"Could not find `{llama_zip_bin}` on PATH. Install llama-zip and ensure the "
            "CLI is available, e.g. `pip3 install .` in the llama-zip repo."
        )

    if not model_path:
        raise LlamaZipModelPathError(
            "Missing model path. Provide `model_path=...` or set ITEXT2BIN_MODEL_PATH."
        )

    # CLI: llama-zip <llm_path> [options] <mode> [input]
    # Use stdin to avoid quoting/length issues.
    cmd = [
        exe,
        model_path,
        "-f",
        options.compressed_format,
        "--n-ctx",
        str(options.n_ctx),
        "-w",
        str(options.window_overlap),
        "--n-gpu-layers",
        str(options.n_gpu_layers),
        "-c",
    ]

    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "llama-zip compression failed.\n"
            f"Command: {cmd!r}\n"
            f"Exit code: {proc.returncode}\n"
            f"stderr:\n{proc.stderr.decode('utf-8', 'replace')}"
        )

    out = proc.stdout.strip()
    # Validate it is base64.
    base64.b64decode(out, validate=True)
    return out


def IText2Bin(
    text: str,
    *,
    model_path: str | None = None,
    options: LlamaZipOptions | None = None,
    llama_zip_bin: str = "llama-zip",
) -> bytes:
    """
    Compress `text` using llama-zip (Q4_K_M model expected) and return base64 bytes.

    Behavior:
    - Compresses using llama-zip with `--n-ctx 8192 -w 25%` (defaults), output format base64.
    - Decodes that base64 to raw compressed bytes.
    - Prepends a 1-byte header containing the length (in bytes) of the compressed payload.
    - Re-encodes header+payload as base64 and returns it as ASCII `bytes`.

    Model:
    - Pass `model_path=...` (path to your Q4_K_M `.gguf`), or set `ITEXT2BIN_MODEL_PATH`.
    """

    if options is None:
        options = LlamaZipOptions()

    if model_path is None:
        model_path = os.environ.get("ITEXT2BIN_MODEL_PATH")

    compressed_b64 = _run_llama_zip_compress_base64(
        text=text,
        model_path=model_path or "",
        options=options,
        llama_zip_bin=llama_zip_bin,
    )
    payload = base64.b64decode(compressed_b64, validate=True)
    framed = _frame_with_1byte_len_header(payload)
    return base64.b64encode(framed)

