#!/usr/bin/env python3
import argparse
import json
import os
import platform
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple


SCRIPT_VERSION = "1"
BASE_URL = os.environ.get("MODEL_SERVICE_BASE_URL", "http://127.0.0.1:8001/v1")
API_KEY = os.environ.get("MODEL_SERVICE_API_KEY", "change-me")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen3-Embedding-4B")
DEFAULT_TIMEOUT = 120
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_ENCODING_FORMAT = "float"
DEFAULT_REPORT_PREFIX = "model_service_embedding_report"


@dataclass
class EnvironmentSnapshot:
    hostname: str
    platform: str
    python_version: str


@dataclass
class ModelEntry:
    model_id: str
    object_type: str


@dataclass
class EmbeddingCase:
    name: str
    inputs: List[str]


@dataclass
class EmbeddingResult:
    case_name: str
    success: bool
    http_status: Optional[int]
    error: str
    elapsed_ms: Optional[float]
    embedding_count: int
    dimension: int
    prompt_tokens: Optional[int]
    total_tokens: Optional[int]
    indices: List[int]
    preview_first8: List[float]


CASES = [
    EmbeddingCase(name="single_short", inputs=["你好，给这句话生成一个 embedding。"]),
    EmbeddingCase(
        name="batch_mixed",
        inputs=[
            "这是第一条文本。",
            "This is the second sample sentence.",
            "搜索召回通常会先做粗排再做精排。",
        ],
    ),
]


def http_request(
    url: str,
    api_key: str,
    method: str = "GET",
    body: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[Optional[int], bytes, Optional[str]]:
    data = None
    headers = {"Authorization": f"Bearer {api_key}"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), None
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        return exc.code, payload, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return None, b"", f"URL error: {exc}"
    except Exception as exc:
        return None, b"", f"Unexpected error: {exc}"


def collect_environment_snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=sys.version.replace("\n", " "),
    )


def fetch_health(base_url: str, api_key: str, timeout: int) -> str:
    root_url = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
    status, payload, error = http_request(f"{root_url}/health", api_key=api_key, timeout=timeout)
    if status is None or error:
      return error or "unknown error"
    text = payload.decode("utf-8", errors="replace").strip()
    return text or f"HTTP {status}"


def fetch_models(base_url: str, api_key: str, timeout: int) -> List[ModelEntry]:
    status, payload, error = http_request(f"{base_url.rstrip('/')}/models", api_key=api_key, timeout=timeout)
    if status != 200:
        raise RuntimeError(error or payload.decode("utf-8", errors="replace") or "failed to fetch models")
    data = json.loads(payload.decode("utf-8"))
    items = data.get("data", [])
    return [ModelEntry(model_id=item.get("id", ""), object_type=item.get("object", "")) for item in items]


def run_embedding_case(
    base_url: str,
    api_key: str,
    model_name: str,
    encoding_format: str,
    timeout: int,
    case: EmbeddingCase,
) -> EmbeddingResult:
    request_body = {
        "model": model_name,
        "input": case.inputs,
        "encoding_format": encoding_format,
    }
    started = time.time()
    status, payload, error = http_request(
        f"{base_url.rstrip('/')}/embeddings",
        api_key=api_key,
        method="POST",
        body=request_body,
        timeout=timeout,
    )
    elapsed_ms = (time.time() - started) * 1000.0

    if status != 200:
        return EmbeddingResult(
            case_name=case.name,
            success=False,
            http_status=status,
            error=error or payload.decode("utf-8", errors="replace"),
            elapsed_ms=elapsed_ms,
            embedding_count=0,
            dimension=0,
            prompt_tokens=None,
            total_tokens=None,
            indices=[],
            preview_first8=[],
        )

    try:
        response = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as exc:
        return EmbeddingResult(
            case_name=case.name,
            success=False,
            http_status=status,
            error=f"Invalid JSON response: {exc}",
            elapsed_ms=elapsed_ms,
            embedding_count=0,
            dimension=0,
            prompt_tokens=None,
            total_tokens=None,
            indices=[],
            preview_first8=[],
        )

    items = response.get("data", [])
    usage = response.get("usage", {})
    first_embedding = items[0].get("embedding", []) if items else []
    preview = [round(float(x), 6) for x in first_embedding[:8]]

    return EmbeddingResult(
        case_name=case.name,
        success=bool(items),
        http_status=status,
        error="" if items else "No embeddings returned",
        elapsed_ms=elapsed_ms,
        embedding_count=len(items),
        dimension=len(first_embedding),
        prompt_tokens=_safe_int(usage.get("prompt_tokens")),
        total_tokens=_safe_int(usage.get("total_tokens")),
        indices=[_safe_int(item.get("index")) or 0 for item in items],
        preview_first8=preview,
    )


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def print_models_summary(models: List[ModelEntry]) -> None:
    print(f"models_count={len(models)}", flush=True)
    for item in models:
        print(f"- id={item.model_id} object={item.object_type}", flush=True)


def print_smoke_summary(results: List[EmbeddingResult]) -> None:
    print("\nEmbedding Smoke Results", flush=True)
    print("case\tsuccess\thttp\telapsed_ms\tembeddings\tdimension\tprompt_tokens\ttotal_tokens", flush=True)
    for row in results:
        print(
            "\t".join(
                [
                    row.case_name,
                    "Y" if row.success else "N",
                    str(row.http_status or ""),
                    f"{row.elapsed_ms:.1f}" if row.elapsed_ms is not None else "",
                    str(row.embedding_count),
                    str(row.dimension),
                    str(row.prompt_tokens or ""),
                    str(row.total_tokens or ""),
                ]
            ),
            flush=True,
        )


def default_markdown_output(model_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_part = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    if not model_part:
        model_part = "embedding"
    return f"{DEFAULT_REPORT_PREFIX}_{model_part}_{ts}.md"


def render_markdown_report(
    base_url: str,
    model_name: str,
    encoding_format: str,
    timeout: int,
    health_text: str,
    environment: EnvironmentSnapshot,
    models: List[ModelEntry],
    results: List[EmbeddingResult],
) -> str:
    lines: List[str] = []
    lines.append("# Embedding Service Report")
    lines.append("")
    lines.append(f"- Generated At: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- Script Version: `{SCRIPT_VERSION}`")
    lines.append(f"- Hostname: `{environment.hostname}`")
    lines.append(f"- Platform: `{environment.platform}`")
    lines.append(f"- Python: `{environment.python_version}`")
    lines.append("")
    lines.append("## Test Config")
    lines.append("")
    lines.append(f"- Base URL: `{base_url}`")
    lines.append(f"- Model Name: `{model_name}`")
    lines.append(f"- Encoding Format: `{encoding_format}`")
    lines.append(f"- Timeout: `{timeout}s`")
    lines.append("")
    lines.append("## Service Discovery")
    lines.append("")
    lines.append(f"- Health Response: `{health_text}`")
    lines.append(f"- Models Count: `{len(models)}`")
    for item in models:
        lines.append(f"- Model Entry: `id={item.model_id}` `object={item.object_type}`")
    lines.append("")
    lines.append("## Embedding Results")
    lines.append("")
    lines.append("| Case | Success | HTTP | Elapsed ms | Embeddings | Dimension | Prompt Tokens | Total Tokens | Indices | Preview First8 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in results:
        indices = ", ".join(str(x) for x in row.indices)
        preview = ", ".join(f"{x:.6f}" for x in row.preview_first8)
        lines.append(
            f"| {row.case_name} | {'Y' if row.success else 'N'} | {row.http_status or ''} | "
            f"{f'{row.elapsed_ms:.1f}' if row.elapsed_ms is not None else ''} | {row.embedding_count} | "
            f"{row.dimension} | {row.prompt_tokens or ''} | {row.total_tokens or ''} | {indices} | {preview} |"
        )
        if row.error:
            lines.append(f"| {row.case_name} error |  |  |  |  |  |  |  |  | `{row.error}` |")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    all_ok = bool(results) and all(item.success for item in results)
    lines.append(f"- Overall Result: `{'PASS' if all_ok else 'FAIL'}`")
    lines.append(f"- Cases: `{len(results)}`")
    if results:
        elapsed_values = [item.elapsed_ms for item in results if item.elapsed_ms is not None]
        if elapsed_values:
            lines.append(
                f"- Elapsed ms Min/Avg/Max: "
                f"`{min(elapsed_values):.1f}/{sum(elapsed_values)/len(elapsed_values):.1f}/{max(elapsed_values):.1f}`"
            )
        dimensions = sorted({item.dimension for item in results if item.dimension > 0})
        lines.append(f"- Dimensions Seen: `{', '.join(str(x) for x in dimensions) if dimensions else ''}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test and report generator for OpenAI-compatible embedding services.")
    parser.add_argument("--base-url", default=BASE_URL, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default=API_KEY, help="Bearer token")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Embedding model name")
    parser.add_argument("--mode", choices=["smoke", "models", "health"], default="smoke", help="Run mode")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Per request timeout in seconds")
    parser.add_argument("--encoding-format", default=DEFAULT_ENCODING_FORMAT, help="Embedding encoding_format field")
    parser.add_argument("--markdown-output", default="", help="Optional Markdown report file path")
    args = parser.parse_args()

    environment = collect_environment_snapshot()
    print(f"Benchmark target: {args.base_url}", flush=True)
    print(f"Model: {args.model_name}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Timeout: {args.timeout}", flush=True)

    if args.mode == "health":
        health_text = fetch_health(args.base_url, args.api_key, args.timeout)
        print(f"health={health_text}", flush=True)
        return 0

    models: List[ModelEntry] = []
    health_text = fetch_health(args.base_url, args.api_key, args.timeout)
    print(f"health={health_text}", flush=True)

    try:
        models = fetch_models(args.base_url, args.api_key, args.timeout)
    except Exception as exc:
        print(f"Failed to fetch models: {exc}", file=sys.stderr, flush=True)
        return 1

    print_models_summary(models)
    if args.mode == "models":
        return 0

    results = [
        run_embedding_case(
            base_url=args.base_url,
            api_key=args.api_key,
            model_name=args.model_name,
            encoding_format=args.encoding_format,
            timeout=args.timeout,
            case=case,
        )
        for case in CASES
    ]
    print_smoke_summary(results)

    report_path = args.markdown_output or default_markdown_output(args.model_name)
    report = render_markdown_report(
        base_url=args.base_url,
        model_name=args.model_name,
        encoding_format=args.encoding_format,
        timeout=args.timeout,
        health_text=health_text,
        environment=environment,
        models=models,
        results=results,
    )
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\nSaved markdown report to: {report_path}", flush=True)

    return 0 if all(item.success for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
