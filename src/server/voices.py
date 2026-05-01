"""Voice registry: named voices mapped to audio file paths.

At request time, ``prepare_conditionals`` is called with the voice's audio
path. This re-encodes the speaker embedding each time, which adds a small
latency cost but avoids the complexity of snapshotting and restoring
model-internal ``Conditionals`` objects.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from chatterbox.inference import ChatterboxInference

logger = logging.getLogger(__name__)


class VoiceError(ValueError):
    """Raised on bad voice input (unknown name, bad base64, etc.)."""


@dataclass
class Voice:
    name: str
    audio_path: str
    exaggeration: float | None = None


class VoiceRegistry:
    def __init__(self, inference: ChatterboxInference):
        self.inference = inference
        self._voices: dict[str, Voice] = {}

    # --- registration ---

    def load_directory(self, voices_dir: str | Path) -> list[str]:
        """Scan ``voices_dir`` for *.wav files and register each by stem name."""
        path = Path(voices_dir)
        if not path.is_dir():
            logger.warning("voices_dir %s does not exist or is not a directory; skipping", path)
            return []
        loaded = []
        for wav in sorted(path.glob("*.wav")):
            name = wav.stem
            self._voices[name] = Voice(name=name, audio_path=str(wav))
            loaded.append(name)
            logger.info("Registered voice '%s' from %s", name, wav)
        return loaded

    def register_from_base64(self, name: str, audio_b64: str, persist_dir: str | Path | None = None,
                             **kwargs) -> Voice:
        """Decode base64 audio and register as a named voice.

        If ``persist_dir`` is given the wav is saved there so it survives
        restarts. Otherwise it goes into a temp file that lives as long as
        the process.
        """
        blob = _decode_b64(audio_b64)
        if persist_dir is not None:
            dest = Path(persist_dir) / f"{name}.wav"
            dest.write_bytes(blob)
            audio_path = str(dest)
        else:
            fd, audio_path = tempfile.mkstemp(suffix=".wav", prefix=f"voice_{name}_")
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
        voice = Voice(name=name, audio_path=audio_path, **kwargs)
        self._voices[name] = voice
        return voice

    # --- access ---

    def get(self, name: str) -> Voice | None:
        return self._voices.get(name)

    def names(self) -> list[str]:
        return sorted(self._voices.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._voices

    def __len__(self) -> int:
        return len(self._voices)

    # --- apply voice onto the model ---

    def apply(self, voice: Voice, **kwargs) -> None:
        """Run ``prepare_conditionals`` for a named voice.

        Must be called under the per-request model lock.
        """
        merged = {}
        if voice.exaggeration is not None:
            merged["exaggeration"] = voice.exaggeration
        merged.update(kwargs)
        self.inference.prepare_conditionals(voice.audio_path, **merged)

    def apply_b64(self, audio_b64: str, **kwargs) -> None:
        """Decode inline base64 audio and run ``prepare_conditionals``.

        Used for one-shot ``audio_prompt_b64`` requests. The temp file is
        deleted immediately after encoding.
        """
        blob = _decode_b64(audio_b64)
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
            self.inference.prepare_conditionals(path, **kwargs)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


def _decode_b64(audio_b64: str) -> bytes:
    try:
        return base64.b64decode(audio_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise VoiceError(f"audio_b64 is not valid base64: {exc}") from exc
