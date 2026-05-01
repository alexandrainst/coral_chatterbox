from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from chatterbox.inference import ChatterboxInference

from server import audio
from server.config import ServerConfig
from server.voices import VoiceError, VoiceRegistry

logger = logging.getLogger(__name__)


# --- request schema ---


class SpeechRequest(BaseModel):
    """OpenAI-compatible body. Unknown keys are accepted (the OpenAI SDK forwards
    ``extra_body`` fields verbatim)."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(default="coral-chatterbox")
    input: str
    voice: Optional[str] = None
    response_format: Literal["wav", "pcm"] = "wav"
    speed: float = 1.0
    stream_format: Optional[Literal["audio", "sse"]] = None  # OpenAI uses None|"audio"|"sse"

    # --- non-standard extras (forwarded via extra_body) ---
    language: Optional[str] = None
    audio_prompt_b64: Optional[str] = None
    exaggeration: Optional[float] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    cfg_weight: Optional[float] = None
    repetition_penalty: Optional[float] = None
    min_p: Optional[float] = None
    max_new_tokens: Optional[int] = None
    normalize_text: Optional[bool] = None
    sentence_split: Optional[bool] = None
    inter_sentence_silence_ms: Optional[int] = None


class RegisterVoiceRequest(BaseModel):
    name: str
    audio_b64: str
    exaggeration: Optional[float] = None


# --- helpers ---


def _media_type_for(fmt: str, sample_rate: int) -> str:
    if fmt == "wav":
        return "audio/wav"
    if fmt == "pcm":
        return f"audio/L16; rate={sample_rate}; channels=1"
    raise HTTPException(status_code=400, detail=f"Unsupported response_format: {fmt}")


def _generation_kwargs(req: SpeechRequest) -> dict[str, Any]:
    """Pick request fields that map to ChatterboxInference.generate kwargs.

    The wrapper itself filters by signature, so unknown keys for a given variant
    are silently dropped with a warning."""
    out: dict[str, Any] = {}
    if req.exaggeration is not None:
        out["exaggeration"] = req.exaggeration
    if req.temperature is not None:
        out["temperature"] = req.temperature
    if req.top_p is not None:
        out["top_p"] = req.top_p
    if req.cfg_weight is not None:
        out["cfg_weight"] = req.cfg_weight
    if req.repetition_penalty is not None:
        out["repetition_penalty"] = req.repetition_penalty
    if req.min_p is not None:
        out["min_p"] = req.min_p
    if req.max_new_tokens is not None:
        out["max_new_tokens"] = req.max_new_tokens
    return out


# --- app factory ---


def build_app(config: ServerConfig) -> FastAPI:
    app = FastAPI(title="coral-chatterbox", version="0.1.0")
    app.state.config = config
    app.state.ready = False
    app.state.inference = None
    app.state.registry = None
    app.state.lock = asyncio.Lock()

    auth_scheme = HTTPBearer(auto_error=False)

    def require_auth(creds: HTTPAuthorizationCredentials | None = Depends(auth_scheme)) -> None:
        if not config.api_key:
            return
        if creds is None or creds.scheme.lower() != "bearer" or creds.credentials != config.api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    @app.on_event("startup")
    def _load() -> None:
        logger.info("Loading model variant=%s repo_id=%s model_dir=%s device=%s",
                    config.variant, config.repo_id, config.model_dir, config.device)
        if config.model_dir:
            inference = ChatterboxInference.from_local(
                ckpt_dir=config.model_dir,
                model_type=config.variant,
                language=config.default_language,
                device=config.device,
            )
        else:
            inference = ChatterboxInference.from_pretrained(
                model_type=config.variant,
                language=config.default_language,
                device=config.device,
                repo_id=config.repo_id,
            )
        registry = VoiceRegistry(inference)
        loaded = registry.load_directory(config.voices_dir)
        logger.info("Loaded %d voices from %s: %s", len(loaded), config.voices_dir, loaded)

        if config.default_voice and config.default_voice not in registry:
            logger.warning("default_voice=%s not found in registry", config.default_voice)
        elif not config.default_voice and loaded:
            config.default_voice = loaded[0]
            logger.info("Using '%s' as default voice", config.default_voice)

        app.state.inference = inference
        app.state.registry = registry
        app.state.ready = True

    # --- endpoints ---

    @app.get("/healthz")
    def healthz() -> Response:
        if not app.state.ready:
            return JSONResponse({"status": "loading"}, status_code=503)
        return JSONResponse({
            "status": "ok",
            "variant": config.variant,
            "device": getattr(app.state.inference.model, "device", "unknown"),
            "voices": app.state.registry.names(),
            "default_voice": config.default_voice,
            "default_language": config.default_language,
            "fast": config.fast,
        })

    @app.get("/v1/models")
    def list_models(_: None = Depends(require_auth)) -> dict:
        return {
            "object": "list",
            "data": [{
                "id": config.model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "coral-chatterbox",
            }],
        }

    @app.post("/v1/audio/voices")
    def register_voice(body: RegisterVoiceRequest, _: None = Depends(require_auth)) -> dict:
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="server warming up")
        kwargs = {}
        if body.exaggeration is not None:
            kwargs["exaggeration"] = body.exaggeration
        try:
            voice = app.state.registry.register_from_base64(body.name, body.audio_b64, **kwargs)
        except VoiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"id": voice.name, "expires_at": None}

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest, _: None = Depends(require_auth)):
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="server warming up")
        if not req.input.strip():
            raise HTTPException(status_code=400, detail="`input` must be non-empty")

        inference: ChatterboxInference = app.state.inference
        registry: VoiceRegistry = app.state.registry

        # Resolve voice. audio_prompt_b64 takes precedence; falls through to
        # library lookup; falls back to default_voice.
        voice = None
        if req.audio_prompt_b64:
            try:
                voice = registry.register_from_base64(
                    name=f"_inline_{int(time.time() * 1000)}",
                    audio_b64=req.audio_prompt_b64,
                    **({"exaggeration": req.exaggeration} if req.exaggeration is not None else {}),
                )
            except VoiceError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            name = req.voice or config.default_voice
            if name is None:
                raise HTTPException(
                    status_code=400,
                    detail="no voice specified and no default voice configured",
                )
            voice = registry.get(name)
            if voice is None:
                raise HTTPException(status_code=404, detail=f"unknown voice '{name}'")

        language = req.language or config.default_language
        gen_kwargs = _generation_kwargs(req)

        # Wrapper-level kwargs accepted by ChatterboxInference.generate /
        # generate_fast / streaming variants. ``sentence_split`` is dropped on
        # the streaming path (always split there).
        wrapper_kwargs: dict[str, Any] = {"language_id": language}
        if req.normalize_text is not None:
            wrapper_kwargs["normalize_text"] = req.normalize_text
        if req.inter_sentence_silence_ms is not None:
            wrapper_kwargs["inter_sentence_silence_ms"] = req.inter_sentence_silence_ms

        media_type = _media_type_for(req.response_format, inference.sr)
        streaming = req.stream_format == "audio"
        use_fast = config.fast and hasattr(inference.model, "generate_fast")

        if streaming:
            async def body_iter():
                async with app.state.lock:
                    registry.apply(voice)
                    stream_fn = (
                        inference.generate_stream_fast_async
                        if use_fast
                        else inference.generate_stream_async
                    )
                    if req.response_format == "wav":
                        yield audio.streaming_wav_header(inference.sr)
                    async for chunk in stream_fn(req.input, **wrapper_kwargs, **gen_kwargs):
                        yield audio.encode_pcm16(chunk)

            return StreamingResponse(body_iter(), media_type=media_type)

        # Blocking path — sentence_split is wrapper-level here.
        if req.sentence_split is not None:
            wrapper_kwargs["sentence_split"] = req.sentence_split

        async with app.state.lock:
            registry.apply(voice)

            def _run():
                if use_fast:
                    return inference.generate_fast(req.input, **wrapper_kwargs, **gen_kwargs)
                return inference.generate(req.input, **wrapper_kwargs, **gen_kwargs)

            wav = await asyncio.to_thread(_run)

        if req.response_format == "wav":
            payload = audio.encode_wav(wav, inference.sr)
        else:
            payload = audio.encode_pcm16(wav)
        return Response(content=payload, media_type=media_type)

    return app
