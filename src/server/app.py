from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from chatterbox.inference import ChatterboxInference

from server import audio
from server.config import ServerConfig
from server.voices import VoiceError, VoiceRegistry

logger = logging.getLogger(__name__)

# Roughly 12MB of decoded audio (base64 inflates by 4/3). Generous for
# reference clips, tight enough to reject obvious abuse.
MAX_AUDIO_B64_BYTES = 16 * 1024 * 1024


# --- request schema ---


class SpeechRequest(BaseModel):
    """OpenAI-compatible body. Unknown keys are accepted (the OpenAI SDK forwards
    ``extra_body`` fields verbatim).

    For streaming, prefer ``response_format=pcm`` — the streaming WAV header
    uses a max-size sentinel that some strict decoders reject.
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str
    voice: str | None = None
    response_format: Literal["wav", "pcm"] = "wav"
    speed: float = 1.0
    stream_format: Literal["audio", "sse"] | None = None  # OpenAI uses None|"audio"|"sse"

    # --- non-standard extras (forwarded via extra_body) ---
    language: str | None = None
    audio_prompt_b64: str | None = None
    exaggeration: float | None = None
    temperature: float | None = None
    top_p: float | None = None
    cfg_weight: float | None = None
    repetition_penalty: float | None = None
    min_p: float | None = None
    max_new_tokens: int | None = None
    normalize_text: bool | None = None
    sentence_split: bool | None = None
    inter_sentence_silence_ms: int | None = None


class RegisterVoiceRequest(BaseModel):
    name: str
    audio_b64: str
    exaggeration: float | None = None


# --- helpers ---


def _media_type_for(fmt: str, sample_rate: int) -> str:
    if fmt == "wav":
        return "audio/wav"
    if fmt == "pcm":
        return f"audio/L16; rate={sample_rate}; channels=1"
    raise HTTPException(status_code=400, detail=f"Unsupported response_format: {fmt}")


def _check_audio_b64_size(audio_b64: str) -> None:
    if len(audio_b64) > MAX_AUDIO_B64_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"audio_b64 exceeds {MAX_AUDIO_B64_BYTES} bytes",
        )


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
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Loading model variant=%s repo_id=%s model_dir=%s device=%s",
            config.variant, config.repo_id, config.model_dir, config.device,
        )
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

        default_voice = config.default_voice
        if default_voice and default_voice not in registry:
            logger.warning("default_voice=%s not found in registry", default_voice)
            default_voice = None
        if not default_voice and loaded:
            default_voice = loaded[0]
            logger.info("Using '%s' as default voice", default_voice)

        if config.api_key is None:
            logger.warning("API_KEY not set — server is unauthenticated")

        app.state.inference = inference
        app.state.registry = registry
        app.state.default_voice = default_voice
        app.state.lock = asyncio.Lock()
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False

    app = FastAPI(title="coral-chatterbox", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.state.ready = False
    app.state.inference = None
    app.state.registry = None
    app.state.default_voice = None

    auth_scheme = HTTPBearer(auto_error=False)

    def require_auth(creds: HTTPAuthorizationCredentials | None = Depends(auth_scheme)) -> None:
        if not config.api_key:
            return
        if creds is None or creds.scheme.lower() != "bearer" or creds.credentials != config.api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

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
            "default_voice": app.state.default_voice,
            "default_language": config.default_language,
            "fast": config.fast,
        })

    @app.get("/v1/models")
    def list_models(_: None = Depends(require_auth)) -> dict:
        import time
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
    async def register_voice(body: RegisterVoiceRequest, _: None = Depends(require_auth)) -> dict:
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="server warming up")
        _check_audio_b64_size(body.audio_b64)
        kwargs: dict[str, Any] = {}
        if body.exaggeration is not None:
            kwargs["exaggeration"] = body.exaggeration
        try:
            voice = app.state.registry.register_from_base64(
                body.name, body.audio_b64,
                persist_dir=config.voices_dir,
                **kwargs,
            )
        except VoiceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("register_from_base64 failed for '%s'", body.name)
            raise HTTPException(status_code=400, detail="could not decode reference audio") from exc
        return {"id": voice.name, "expires_at": None}

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest, _: None = Depends(require_auth)):
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="server warming up")
        if not req.input.strip():
            raise HTTPException(status_code=400, detail="`input` must be non-empty")
        if req.audio_prompt_b64 is not None:
            _check_audio_b64_size(req.audio_prompt_b64)
        if req.speed != 1.0:
            logger.warning(
                "speed=%s requested but ignored — Chatterbox has no time-stretch control",
                req.speed,
            )

        inference: ChatterboxInference = app.state.inference
        registry: VoiceRegistry = app.state.registry

        # Resolve the voice reference (registry lookup or inline b64 check).
        voice = None
        if not req.audio_prompt_b64:
            name = req.voice or app.state.default_voice
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

        wrapper_kwargs: dict[str, Any] = {"language_id": language}
        if req.normalize_text is not None:
            wrapper_kwargs["normalize_text"] = req.normalize_text
        if req.inter_sentence_silence_ms is not None:
            wrapper_kwargs["inter_sentence_silence_ms"] = req.inter_sentence_silence_ms

        media_type = _media_type_for(req.response_format, inference.sr)
        streaming = req.stream_format == "audio"
        use_fast = config.fast and hasattr(inference.model, "generate_fast")

        def _prepare_voice():
            """Run prepare_conditionals for the resolved voice. Must be called under lock."""
            cond_kwargs: dict[str, Any] = {}
            if req.exaggeration is not None:
                cond_kwargs["exaggeration"] = req.exaggeration
            if voice is not None:
                registry.apply(voice, **cond_kwargs)
            else:
                try:
                    registry.apply_b64(req.audio_prompt_b64, **cond_kwargs)
                except VoiceError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                except Exception as exc:
                    logger.exception("apply_b64 failed for inline audio_prompt_b64")
                    raise HTTPException(
                        status_code=400, detail="could not decode reference audio",
                    ) from exc

        if streaming:
            async def body_iter():
                async with app.state.lock:
                    _prepare_voice()
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

        if req.sentence_split is not None:
            wrapper_kwargs["sentence_split"] = req.sentence_split

        async with app.state.lock:
            _prepare_voice()

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
