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
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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
        self._voices: Dict[str, Voice] = {}
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
                self._register_from_path(name, str(wav))
                loaded.append(name)
                logger.info("Registered voice '%s' from %s", name, wav)
            except Exception:
                logger.exception("Failed to load voice '%s' from %s", name, wav)
        return loaded

    def register_from_base64(self, name: str, audio_b64: str, **kwargs) -> Voice:
        try:
            blob = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise VoiceError(f"audio_b64 is not valid base64: {exc}") from exc
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(blob)
            tmp.flush()
            return self._register_from_path(name, tmp.name, **kwargs)

    def _register_from_path(self, name: str, path: str, **kwargs) -> Voice:
        # ChatterboxInference.prepare_conditionals filters kwargs to what the
        # underlying variant accepts (exaggeration, norm_loudness, ...).
        self.inference.prepare_conditionals(path, **kwargs)
        conds = self._snapshot_conds()
        if conds is None:
            raise VoiceError(f"prepare_conditionals produced no conds for '{name}'")
        voice = Voice(name=name, conds=conds)
        with self._lock:
            self._voices[name] = voice
        # Reset the wrapper cache so the next request doesn't think a stale path
        # is still loaded.
        self.inference._last_audio_prompt_path = None
        return voice

    # --- access ---

    def get(self, name: str) -> Optional[Voice]:
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
        self.inference._last_audio_prompt_path = f"<voice:{voice.name}>"

    def _snapshot_conds(self):
        """Deep-copy the current ``model.conds`` so subsequent calls don't mutate it."""
        conds = getattr(self.inference.model, "conds", None)
        if conds is None:
            return None
        try:
            return copy.deepcopy(conds)
        except Exception:
            # Fall back to a shallow copy on the dataclass; tensors stay shared.
            # This is rare — the Conditionals dataclasses contain tensors that
            # deepcopy handles fine.
            logger.exception("Deep-copy of conds failed; using the live object directly")
            return conds
