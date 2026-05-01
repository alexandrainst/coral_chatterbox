"""Tensor → audio bytes encoders for the speech endpoint.

Only WAV (16-bit PCM) and raw PCM16 LE are supported. Both formats are
produced from the float32 model output without external dependencies.
"""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
import torch


def _to_pcm16_bytes(wav: torch.Tensor) -> bytes:
    """Convert a float32 audio tensor (any shape) to raw 16-bit PCM little-endian bytes."""
    array = wav.detach().squeeze().cpu().to(torch.float32).numpy()
    array = np.clip(array, -1.0, 1.0)
    pcm = (array * 32767.0).astype("<i2", copy=False)
    return pcm.tobytes()


def encode_wav(wav: torch.Tensor, sample_rate: int) -> bytes:
    """Encode a tensor as a complete WAV file (mono, 16-bit PCM)."""
    pcm = _to_pcm16_bytes(wav)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def encode_pcm16(wav: torch.Tensor) -> bytes:
    """Encode a tensor as raw 16-bit PCM little-endian (no header)."""
    return _to_pcm16_bytes(wav)


def streaming_wav_header(sample_rate: int) -> bytes:
    """RIFF/WAVE header for a streaming response of unknown total length.

    Uses 0xFFFFFFFF in the size fields, which most decoders (ffmpeg, vlc, mpv,
    Python's ``wave`` in streaming mode) treat as 'read until EOF'.
    """
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    header = b"RIFF"
    header += struct.pack("<I", 0xFFFFFFFF)  # file size - 8
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack("<I", 16)  # fmt chunk size
    header += struct.pack("<H", 1)  # PCM
    header += struct.pack("<H", num_channels)
    header += struct.pack("<I", sample_rate)
    header += struct.pack("<I", byte_rate)
    header += struct.pack("<H", block_align)
    header += struct.pack("<H", bits_per_sample)
    header += b"data"
    header += struct.pack("<I", 0xFFFFFFFF)  # data chunk size
    return header
