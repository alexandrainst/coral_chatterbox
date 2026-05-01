from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Literal

VARIANTS = ("base", "multilingual", "turbo")


@dataclass
class ServerConfig:
    variant: Literal["base", "multilingual", "turbo"] = "multilingual"
    repo_id: str | None = None
    model_dir: str | None = None
    voices_dir: str = "/voices"
    default_voice: str | None = None
    default_language: str = "en"
    device: str | None = None
    fast: bool = True
    api_key: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    model_id: str = "coral-chatterbox"


def _env(name: str, default=None):
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m server",
        description="OpenAI-compatible HTTP server wrapping ChatterboxInference.",
    )
    p.add_argument("--variant", choices=VARIANTS, default=_env("VARIANT", "multilingual"))
    p.add_argument("--repo-id", default=_env("REPO_ID"))
    p.add_argument("--model-dir", default=_env("MODEL_DIR"))
    p.add_argument("--voices-dir", default=_env("VOICES_DIR", "/voices"))
    p.add_argument("--default-voice", default=_env("DEFAULT_VOICE"))
    p.add_argument("--default-language", default=_env("DEFAULT_LANGUAGE", "en"))
    p.add_argument("--device", default=_env("DEVICE"))
    p.add_argument(
        "--no-fast",
        dest="fast",
        action="store_false",
        default=not _env_bool("NO_FAST", False),
        help="Disable the CUDA-graph fast path (generate_fast).",
    )
    p.add_argument("--api-key", default=_env("API_KEY"))
    p.add_argument("--host", default=_env("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(_env("PORT", "8000")))
    p.add_argument("--model-id", default=_env("MODEL_ID", "coral-chatterbox"))
    return p


def config_from_args(argv=None) -> ServerConfig:
    args = build_parser().parse_args(argv)
    return ServerConfig(
        variant=args.variant,
        repo_id=args.repo_id,
        model_dir=args.model_dir,
        voices_dir=args.voices_dir,
        default_voice=args.default_voice,
        default_language=args.default_language,
        device=args.device,
        fast=args.fast,
        api_key=args.api_key,
        host=args.host,
        port=args.port,
        model_id=args.model_id,
    )
