"""Voice registry: precomputed Conditionals indexed by name.

A ``ChatterboxInference`` instance only holds one set of speaker conditionals
(``inference.model.conds``). To avoid re-encoding reference audio on every
request, we run ``prepare_conditionals`` once per voice file at startup, then
deep-copy the resulting ``Conditionals`` object into a dict. At request time
we swap the cached object back onto the model under a single asyncio lock.
"""

from __future__ import annotations

import base64
import binascii
import copy
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from chatterbox.inference import ChatterboxInference

logger = logging.getLogger(__name__)


class VoiceError(ValueError):
    """Raised on bad voice input (unknown name, bad base64, etc.)."""


@dataclass
class Voice:
    name: str
    conds: object  # chatterbox Conditionals dataclass; type varies per variant


class VoiceRegistry:
    def __init__(self, inference: ChatterboxInference):
        self.inference = inference
        self._voices: dict[str, Voice] = {}
        self._lock = threading.Lock()

    # --- registration ---

    def load_directory(self, voices_dir: str | Path) -> list[str]:
        """Scan ``voices_dir`` for *.wav files and precompute conditionals for each."""
        path = Path(voices_dir)
        if not path.is_dir():
            logger.warning("voices_dir %s does not exist or is not a directory; skipping", path)
            return []
        loaded = []
        for wav in sorted(path.glob("*.wav")):
            name = wav.stem
            try:
                self._register_named(name, str(wav))
                loaded.append(name)
                logger.info("Registered voice '%s' from %s", name, wav)
            except Exception:
                logger.exception("Failed to load voice '%s' from %s", name, wav)
        return loaded

    def register_from_base64(self, name: str, audio_b64: str, **kwargs) -> Voice:
        """Decode base64, run prepare_conditionals, store under ``name``."""
        with _b64_to_tempfile(audio_b64) as path:
            return self._register_named(name, path, **kwargs)

    def register_transient(self, audio_b64: str, **kwargs) -> Voice:
        """Like ``register_from_base64`` but does not retain the voice in the
        registry — used for one-shot inline prompts where we don't want to
        accumulate entries across requests."""
        with _b64_to_tempfile(audio_b64) as path:
            conds = self._compute_conds(path, **kwargs)
        return Voice(name="<transient>", conds=conds)

    def _register_named(self, name: str, path: str, **kwargs) -> Voice:
        conds = self._compute_conds(path, **kwargs)
        voice = Voice(name=name, conds=conds)
        with self._lock:
            self._voices[name] = voice
        return voice

    def _compute_conds(self, path: str, **kwargs):
        # ChatterboxInference.prepare_conditionals filters kwargs to what the
        # underlying variant accepts (exaggeration, norm_loudness, ...).
        self.inference.prepare_conditionals(path, **kwargs)
        conds = getattr(self.inference.model, "conds", None)
        if conds is None:
            raise VoiceError(f"prepare_conditionals produced no conds for '{path}'")
        snapshot = copy.deepcopy(conds)
        # Reset the wrapper cache so the next request doesn't think a stale
        # path is still loaded.
        self.inference.invalidate_prompt_cache()
        return snapshot

    # --- access ---

    def get(self, name: str) -> Voice | None:
        with self._lock:
            return self._voices.get(name)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._voices.keys())

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._voices

    def __len__(self) -> int:
        with self._lock:
            return len(self._voices)

    # --- swap onto the model ---

    def apply(self, voice: Voice) -> None:
        """Swap a cached voice's conditionals onto the inference model.

        Must be called under the per-process model lock. Sets a sentinel cache
        marker so ``ChatterboxInference.generate`` does not re-encode.
        """
        self.inference.model.conds = voice.conds
        # Sentinel string keeps the wrapper's cache check happy without
        # pointing at a real path.
        self.inference._last_audio_prompt_path = f"<voice:{voice.name}>"


class _b64_to_tempfile:
    """Context manager: decode base64 into a temp .wav, yield its path,
    delete on exit. Portable across platforms (avoids the Windows
    NamedTemporaryFile reopen restriction)."""

    def __init__(self, audio_b64: str):
        self.audio_b64 = audio_b64
        self.path: str | None = None

    def __enter__(self) -> str:
        try:
            blob = base64.b64decode(self.audio_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise VoiceError(f"audio_b64 is not valid base64: {exc}") from exc
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
        except Exception:
            os.unlink(path)
            raise
        self.path = path
        return path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path is not None:
            try:
                os.unlink(self.path)
            except OSError:
                pass
