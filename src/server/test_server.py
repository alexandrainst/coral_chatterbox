#!/usr/bin/env python3
"""Smoke-test script for the coral-chatterbox OpenAI-compatible server.

Usage:
    # Start the server first, then:
    python test_server.py                          # defaults: localhost:8000, no auth
    python test_server.py --base-url http://host:9000 --api-key sk-test

    # Only run specific tests:
    python test_server.py --only healthz,models,speech

    # With a voice directory pre-loaded on the server:
    python test_server.py --voice <name>           # use a named voice for speech tests

Requires: httpx (pip install httpx) — no dependency on the server code.
"""

from __future__ import annotations

import argparse
import base64
import struct
import sys
import time
import wave
import io

try:
    import httpx
except ImportError:
    sys.exit("httpx is required: pip install httpx")


def make_silent_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a silent mono 16-bit WAV in memory."""
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


SILENT_WAV_B64 = base64.b64encode(make_silent_wav()).decode()

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


class ServerTester:
    def __init__(self, base_url: str, api_key: str | None, voice: str | None, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.voice = voice
        self.timeout = timeout
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)
        self.results: list[tuple[str, bool, str]] = []

    def _record(self, name: str, passed: bool, detail: str = ""):
        tag = PASS if passed else FAIL
        line = f"  {tag}  {name}"
        if detail:
            line += f"  ({detail})"
        print(line)
        self.results.append((name, passed, detail))

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_healthz(self):
        print("\n--- /healthz ---")
        r = self.client.get("/healthz")
        self._record("GET /healthz status", r.status_code == 200, f"status={r.status_code}")
        if r.status_code == 200:
            body = r.json()
            self._record("has 'status' field", body.get("status") == "ok", f"status={body.get('status')}")
            self._record("has 'variant' field", "variant" in body, f"variant={body.get('variant')}")
            self._record("has 'voices' field", isinstance(body.get("voices"), list),
                         f"voices={body.get('voices')}")
            if self.voice is None and body.get("default_voice"):
                self.voice = body["default_voice"]
                print(f"  (auto-selected voice: {self.voice})")

    def test_models(self):
        print("\n--- /v1/models ---")
        r = self.client.get("/v1/models")
        self._record("GET /v1/models status", r.status_code == 200, f"status={r.status_code}")
        if r.status_code == 200:
            body = r.json()
            data = body.get("data", [])
            self._record("returns model list", len(data) > 0, f"count={len(data)}")
            if data:
                self._record("model has 'id'", "id" in data[0], f"id={data[0].get('id')}")

    def test_speech_missing_input(self):
        print("\n--- /v1/audio/speech (validation) ---")
        r = self.client.post("/v1/audio/speech", json={"input": "", "voice": "x"})
        self._record("empty input → 400", r.status_code == 400, f"status={r.status_code}")

    def test_speech_unknown_voice(self):
        r = self.client.post("/v1/audio/speech", json={"input": "hello", "voice": "__nonexistent__"})
        self._record("unknown voice → 404", r.status_code == 404, f"status={r.status_code}")

    def test_speech_no_voice(self):
        r = self.client.post("/v1/audio/speech", json={"input": "hello"})
        expected = 400 if self.voice is None else 200
        ok = r.status_code == expected
        self._record(f"no voice field → {expected}", ok, f"status={r.status_code}")

    def test_register_voice(self):
        print("\n--- /v1/audio/voices (register) ---")
        r = self.client.post("/v1/audio/voices", json={
            "name": "__test_voice__",
            "audio_b64": SILENT_WAV_B64,
        })
        self._record("register voice", r.status_code == 200, f"status={r.status_code}")
        if r.status_code == 200:
            body = r.json()
            self._record("returns id", body.get("id") == "__test_voice__", f"id={body.get('id')}")

    def test_register_voice_bad_b64(self):
        r = self.client.post("/v1/audio/voices", json={
            "name": "__bad__",
            "audio_b64": "not-valid-base64!!!",
        })
        self._record("bad base64 → 400", r.status_code == 400, f"status={r.status_code}")

    def test_speech_wav(self):
        print("\n--- /v1/audio/speech (generate) ---")
        voice = self.voice
        if not voice:
            print(f"  {SKIP}  speech_wav — no voice available (use --voice or preload voices)")
            return

        start = time.monotonic()
        r = self.client.post("/v1/audio/speech", json={
            "input": "Hello, this is a test.",
            "voice": voice,
            "response_format": "wav",
        })
        elapsed = time.monotonic() - start
        self._record("speech WAV status", r.status_code == 200, f"status={r.status_code} {elapsed:.1f}s")
        if r.status_code == 200:
            self._record("WAV content-type", "audio/wav" in r.headers.get("content-type", ""),
                         r.headers.get("content-type", ""))
            self._record("WAV body > 44 bytes", len(r.content) > 44, f"size={len(r.content)}")
            ok = r.content[:4] == b"RIFF" and r.content[8:12] == b"WAVE"
            self._record("valid RIFF/WAVE header", ok)

    def test_speech_pcm(self):
        voice = self.voice
        if not voice:
            print(f"  {SKIP}  speech_pcm — no voice available")
            return

        r = self.client.post("/v1/audio/speech", json={
            "input": "PCM test.",
            "voice": voice,
            "response_format": "pcm",
        })
        self._record("speech PCM status", r.status_code == 200, f"status={r.status_code}")
        if r.status_code == 200:
            self._record("PCM content-type", "audio/L16" in r.headers.get("content-type", ""),
                         r.headers.get("content-type", ""))
            self._record("PCM body > 0 bytes", len(r.content) > 0, f"size={len(r.content)}")

    def test_speech_inline_b64(self):
        print("\n--- /v1/audio/speech (inline audio_prompt_b64) ---")
        r = self.client.post("/v1/audio/speech", json={
            "input": "Inline voice test.",
            "audio_prompt_b64": SILENT_WAV_B64,
            "response_format": "wav",
        })
        self._record("inline b64 status", r.status_code == 200, f"status={r.status_code}")

    def test_speech_streaming(self):
        print("\n--- /v1/audio/speech (streaming) ---")
        voice = self.voice
        if not voice:
            print(f"  {SKIP}  streaming — no voice available")
            return

        with self.client.stream("POST", "/v1/audio/speech", json={
            "input": "Streaming test sentence.",
            "voice": voice,
            "response_format": "pcm",
            "stream_format": "audio",
        }) as r:
            self._record("streaming status", r.status_code == 200, f"status={r.status_code}")
            if r.status_code == 200:
                chunks = list(r.iter_bytes(chunk_size=4096))
                total = sum(len(c) for c in chunks)
                self._record("received chunks", len(chunks) > 0, f"chunks={len(chunks)} total={total}B")

    def test_audio_b64_too_large(self):
        print("\n--- size limit ---")
        big = base64.b64encode(b"\x00" * (17 * 1024 * 1024)).decode()
        r = self.client.post("/v1/audio/voices", json={
            "name": "__toobig__",
            "audio_b64": big,
        })
        self._record("oversized b64 → 413", r.status_code == 413, f"status={r.status_code}")

    def test_auth_required(self):
        """Only meaningful if server was started with --api-key."""
        print("\n--- auth ---")
        no_auth = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        r = no_auth.get("/v1/models")
        if r.status_code == 401:
            self._record("no auth → 401", True)
        else:
            print(f"  {SKIP}  auth check — server appears unauthenticated (status={r.status_code})")

    # ------------------------------------------------------------------

    def run(self, only: set[str] | None = None):
        all_tests = [
            ("healthz", self.test_healthz),
            ("models", self.test_models),
            ("validation", self.test_speech_missing_input),
            ("unknown_voice", self.test_speech_unknown_voice),
            ("no_voice", self.test_speech_no_voice),
            ("register", self.test_register_voice),
            ("register_bad", self.test_register_voice_bad_b64),
            ("speech", self.test_speech_wav),
            ("pcm", self.test_speech_pcm),
            ("inline", self.test_speech_inline_b64),
            ("streaming", self.test_speech_streaming),
            ("size_limit", self.test_audio_b64_too_large),
            ("auth", self.test_auth_required),
        ]

        for name, fn in all_tests:
            if only and name not in only:
                continue
            try:
                fn()
            except httpx.ConnectError:
                print(f"\n  {FAIL}  {name} — connection refused (is the server running?)")
                self.results.append((name, False, "connection refused"))
            except Exception as exc:
                print(f"\n  {FAIL}  {name} — {exc}")
                self.results.append((name, False, str(exc)))

        # summary
        passed = sum(1 for _, ok, _ in self.results if ok)
        total = len(self.results)
        failed = total - passed
        print(f"\n{'='*40}")
        print(f"  {passed}/{total} passed", end="")
        if failed:
            print(f", {failed} failed")
            names = [n for n, ok, _ in self.results if not ok]
            print(f"  Failed: {', '.join(names)}")
        else:
            print()
        return failed == 0


def main():
    p = argparse.ArgumentParser(description="Smoke-test the coral-chatterbox server")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--api-key", default=None)
    p.add_argument("--voice", default=None, help="Named voice to use for speech tests")
    p.add_argument("--timeout", type=float, default=120, help="Per-request timeout in seconds")
    p.add_argument("--only", default=None,
                   help="Comma-separated list of tests to run (healthz,models,speech,...)")
    args = p.parse_args()

    only = set(args.only.split(",")) if args.only else None
    tester = ServerTester(args.base_url, args.api_key, args.voice, args.timeout)
    ok = tester.run(only)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
