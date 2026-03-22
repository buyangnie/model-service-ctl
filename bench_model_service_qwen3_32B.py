#!/usr/bin/env python3
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple


SCRIPT_VERSION = "5"
BASE_URL = os.environ.get("MODEL_SERVICE_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("MODEL_SERVICE_API_KEY", "change-me")
MODELS = [
    {
        "name": "Qwen3-32B",
        "tokenizer": "/data/models/Qwen3-32B",
    },
]

DEFAULT_MODE = "all"
DEFAULT_CONTEXT_SIZES = [128, 512, 2048, 4096, 8192]
DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 1800
DEFAULT_FIRST_TOKEN_TIMEOUT = 60
DEFAULT_PROGRESS_INTERVAL = 5
DEFAULT_REPORT_PREFIX = "model_service_benchmark_report"
DEFAULT_TESTSET_PATH = "model_service_testset.json"
DEFAULT_THINKING_MODE = os.environ.get("MODEL_SERVICE_THINKING_MODE", "no_think")


def join_sections(title: str, sections: List[str]) -> str:
    lines = [title]
    for idx, section in enumerate(sections, start=1):
        lines.append(f"\n### 材料{idx}\n{section}")
    return "\n".join(lines)


def build_agentic_planning_case() -> dict:
    sections = [
        "项目目标：为客服知识库问答系统设计一次灰度发布方案。要求先完成资料检索，再核验版本，再决定是否升级索引。",
        "系统约束：不能直接修改生产索引；如果发现文档版本低于 v2.4，则必须先回源同步；输出不能超过三行。",
        "可用工具说明：tool.search_docs 用于检索资料；tool.check_version 用于核验当前索引版本；tool.enqueue_rebuild 用于提交重建任务。",
        "历史事故复盘：之前因为跳过版本核验，导致旧索引直接覆盖了新分片，恢复耗时 43 分钟。",
        "当前环境：客户正在做双十一压测，允许后台重建，但不允许前台中断服务；若资料不完整，应先检索后判断。",
        "补充背景：项目 owner 希望代理先列计划，再执行。计划必须包含检索、核验、决定动作三步。",
    ]
    return {
        "name": "agentic_planning_long_context",
        "description": "长输入下的 agent 任务规划",
        "messages": [
            {"role": "system", "content": "你是一个擅长规划工具调用顺序的中文助手。"},
            {
                "role": "user",
                "content": join_sections(
                    "下面是一次代理任务的完整背景，请阅读所有材料后回答。",
                    sections * 6
                    + [
                        "最终要求：请只输出三行执行计划，每行一个步骤，格式必须是“1. ...”这种编号形式，不要解释。"
                    ],
                ),
            },
        ],
        "max_tokens": 48,
        "contains": ["检索", "核验", "升级"],
    }


def build_rag_json_case() -> dict:
    sections = [
        "知识库文档 A：项目名称“星港客服助手”，负责人“王岚”，当前计划版本 v2.7，目标区域“华东二区”。",
        "知识库文档 B：历史版本 v2.6 的负责人曾是“赵衡”，该信息已经过期，不应再被采用。",
        "变更公告：自 2026-03-15 起，负责人正式变更为“王岚”，部署窗口更新为“周三 02:30-03:00”。",
        "FAQ 摘要：若多个文档冲突，以“变更公告”优先，其次是“知识库文档 A”，最后才看历史文档。",
        "检索日志：本次召回命中文档 A、B 和变更公告，B 的时间戳早于其他两份。"
    ]
    return {
        "name": "rag_structured_json_long_context",
        "description": "多文档 RAG 后返回短 JSON",
        "messages": [
            {"role": "system", "content": "你是一个根据检索结果输出精简 JSON 的中文助手。"},
            {
                "role": "user",
                "content": join_sections(
                    "下面是多文档检索结果，请根据优先级规则给出最终结论。",
                    sections * 8
                    + [
                        "最终问题：请严格输出一个 JSON 对象，不要解释，字段只有"
                        '{"负责人":"","版本":"","部署窗口":""}。'
                    ],
                ),
            },
        ],
        "max_tokens": 48,
        "contains": ['"负责人"', '"王岚"', '"版本"', '"v2.7"', '"部署窗口"', '"周三 02:30-03:00"'],
    }


def build_rag_fact_case() -> dict:
    sections = [
        "文档 1：订单归档服务已迁移到新集群，但接口地址尚未对外公布。",
        "文档 2：检索任务采用分层召回，先走倒排，再走向量补充，最终由 rerank 排序。",
        "文档 3：本次联调使用的发布批次号为 RAG-4271，该字段在后续审批表中也会出现。",
        "文档 4：如果用户只询问批次号，应直接返回批次号本身，不要附加描述。",
        "审批备注：旧批次号 RAG-4188 已废弃，新的有效批次号是 RAG-4271。"
    ]
    return {
        "name": "rag_fact_retrieval_long_context",
        "description": "长输入 RAG 精确事实抽取",
        "messages": [
            {"role": "system", "content": "你是一个擅长从长文档中精确抽取答案的中文助手。"},
            {
                "role": "user",
                "content": join_sections(
                    "下面是一次 RAG 召回结果，请先通读再作答。",
                    sections * 10 + ["问题：请只返回有效发布批次号。"],
                ),
            },
        ],
        "max_tokens": 12,
        "contains": ["RAG-4271"],
    }


SMOKE_CASES = [
    build_agentic_planning_case(),
    build_rag_json_case(),
    build_rag_fact_case(),
]


def apply_thinking_mode(messages: List[dict], thinking_mode: str) -> List[dict]:
    if thinking_mode == "think":
        return messages

    patched = [dict(message) for message in messages]
    injected = False
    for message in patched:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = f"/no_think\n{content}"
            injected = True
            break
    if not injected and patched:
        content = patched[0].get("content", "")
        if isinstance(content, str):
            patched[0]["content"] = f"/no_think\n{content}"
    return patched


@dataclass
class MetricsSnapshot:
    collected: bool
    source: str
    kv_cache_usage_perc: Optional[float] = None
    prefix_cache_queries: Optional[float] = None
    prefix_cache_hits: Optional[float] = None
    gpu_utilization_avg: Optional[float] = None
    gpu_memory_used_mb_total: Optional[float] = None
    gpu_memory_utilization_avg: Optional[float] = None
    raw_notes: List[str] = field(default_factory=list)


@dataclass
class EnvironmentSnapshot:
    hostname: str
    platform: str
    kernel: str
    os_release: str
    python_version: str
    cpu_model: str
    cpu_cores_logical: Optional[int]
    cpu_cores_physical: Optional[int]
    total_memory_gb: Optional[float]
    disk_root_total_gb: Optional[float]
    disk_root_free_gb: Optional[float]
    gpu_summary: List[str]
    nvidia_smi_summary: str
    raw_notes: List[str] = field(default_factory=list)


@dataclass
class StreamResponse:
    success: bool
    http_status: Optional[int]
    error: str
    first_token_timeout_triggered: bool
    ttft_seconds: Optional[float]
    total_seconds: Optional[float]
    decode_seconds: Optional[float]
    raw_chunks: int
    output_text: str
    prompt_tokens_reported: Optional[int]
    completion_tokens_reported: Optional[int]
    total_tokens_reported: Optional[int]


@dataclass
class SmokeResult:
    model: str
    case_name: str
    case_description: str
    success: bool
    http_status: Optional[int]
    error: str
    ttft_seconds: Optional[float]
    total_seconds: Optional[float]
    decode_seconds: Optional[float]
    raw_chunks: int
    prompt_tokens_estimated: int
    completion_tokens_estimated: int
    prompt_tokens_reported: Optional[int]
    completion_tokens_reported: Optional[int]
    total_tokens_reported: Optional[int]
    output_tps_estimated: Optional[float]
    output_text: str
    output_preview: str
    check_passed: bool
    check_message: str
    metrics_before: MetricsSnapshot
    metrics_after: MetricsSnapshot


@dataclass
class BenchResult:
    model: str
    case_name: str
    thinking_mode: str
    context_tokens_target: int
    success: bool
    http_status: Optional[int]
    error: str
    ttft_seconds: Optional[float]
    total_seconds: Optional[float]
    decode_seconds: Optional[float]
    raw_chunks: int
    prompt_tokens_estimated: int
    completion_tokens_estimated: int
    prompt_tokens_reported: Optional[int]
    completion_tokens_reported: Optional[int]
    total_tokens_reported: Optional[int]
    output_tps_estimated: Optional[float]
    output_preview: str
    metrics_before: MetricsSnapshot
    metrics_after: MetricsSnapshot


class TokenEstimator:
    def __init__(self) -> None:
        self._cache: Dict[str, Optional[object]] = {}
        self._status: Dict[str, str] = {}

    def count_text(self, tokenizer_ref: str, text: str) -> int:
        tokenizer = self._get_tokenizer(tokenizer_ref)
        if tokenizer is not None:
            try:
                encoded = tokenizer.encode(text, add_special_tokens=False)
                return len(encoded)
            except Exception:
                pass
        return self._rough_estimate(text)

    def count_messages(self, tokenizer_ref: str, messages: List[dict]) -> int:
        parts = []
        for message in messages:
            role = str(message.get("role", ""))
            content = message.get("content", "")
            if isinstance(content, list):
                normalized = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        normalized.append(item.get("text", ""))
                content_text = "".join(normalized)
            else:
                content_text = str(content)
            parts.append(f"[{role}]\n{content_text}\n")
        return self.count_text(tokenizer_ref, "\n".join(parts))

    def status(self, tokenizer_ref: str) -> str:
        self._get_tokenizer(tokenizer_ref)
        return self._status.get(tokenizer_ref, "rough_estimate")

    def _get_tokenizer(self, tokenizer_ref: str) -> Optional[object]:
        if tokenizer_ref in self._cache:
            return self._cache[tokenizer_ref]

        try:
            from transformers import AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_ref,
                trust_remote_code=True,
            )
            self._cache[tokenizer_ref] = tokenizer
            self._status[tokenizer_ref] = f"transformers:{tokenizer_ref}"
            return tokenizer
        except Exception as exc:
            self._cache[tokenizer_ref] = None
            self._status[tokenizer_ref] = f"rough_estimate ({exc})"
            return None

    @staticmethod
    def _rough_estimate(text: str) -> int:
        if not text:
            return 0
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        latin_words = len(re.findall(r"[A-Za-z0-9_]+", text))
        punct = len(re.findall(r"[，。！？；：、“”‘’（）《》【】,.!?;:()\[\]{}\"'/-]", text))
        return max(1, cjk_chars + latin_words + max(1, punct // 4))


def post_json(url: str, api_key: str, payload: dict) -> urllib.request.Request:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    return req


def extract_stream_text(obj: dict) -> str:
    choices = obj.get("choices") or []
    if not choices:
        return ""

    choice = choices[0] or {}
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)

    text = choice.get("text")
    return text if isinstance(text, str) else ""


def extract_usage(obj: dict) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return None, None, None
    return (
        _safe_int(usage.get("prompt_tokens")),
        _safe_int(usage.get("completion_tokens")),
        _safe_int(usage.get("total_tokens")),
    )


def run_stream_chat(
    base_url: str,
    model: str,
    api_key: str,
    messages: List[dict],
    max_tokens: int,
    temperature: float,
    timeout: int,
    first_token_timeout: int,
    on_first_token: Optional[Callable[[float], None]] = None,
) -> StreamResponse:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    start = time.perf_counter()
    first_token_time = None
    end_time = None
    chunks = 0
    generated_parts: List[str] = []
    http_status: Optional[int] = None
    prompt_tokens_reported: Optional[int] = None
    completion_tokens_reported: Optional[int] = None
    total_tokens_reported: Optional[int] = None

    try:
        req = post_json(url, api_key, payload)
        read_timeout = min(timeout, first_token_timeout) if first_token_timeout > 0 else timeout
        with urllib.request.urlopen(req, timeout=read_timeout) as resp:
            http_status = getattr(resp, "status", None)
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue

                data = line[6:]
                if data == "[DONE]":
                    end_time = time.perf_counter()
                    break

                chunks += 1
                obj = json.loads(data)
                usage_prompt, usage_completion, usage_total = extract_usage(obj)
                prompt_tokens_reported = usage_prompt or prompt_tokens_reported
                completion_tokens_reported = usage_completion or completion_tokens_reported
                total_tokens_reported = usage_total or total_tokens_reported

                text = extract_stream_text(obj)
                if text:
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                        if on_first_token is not None:
                            on_first_token(first_token_time - start)
                    generated_parts.append(text)
                    end_time = now
    except urllib.error.HTTPError as exc:
        return StreamResponse(
            success=False,
            http_status=exc.code,
            error=exc.read().decode("utf-8", errors="replace"),
            first_token_timeout_triggered=False,
            ttft_seconds=None,
            total_seconds=None,
            decode_seconds=None,
            raw_chunks=chunks,
            output_text="",
            prompt_tokens_reported=prompt_tokens_reported,
            completion_tokens_reported=completion_tokens_reported,
            total_tokens_reported=total_tokens_reported,
        )
    except (socket.timeout, TimeoutError) as exc:
        elapsed = time.perf_counter() - start
        first_token_timeout_triggered = first_token_time is None and first_token_timeout > 0 and elapsed >= first_token_timeout
        error = (
            f"first token timeout after {first_token_timeout}s"
            if first_token_timeout_triggered
            else str(exc)
        )
        return StreamResponse(
            success=False,
            http_status=http_status,
            error=error,
            first_token_timeout_triggered=first_token_timeout_triggered,
            ttft_seconds=None,
            total_seconds=elapsed,
            decode_seconds=None,
            raw_chunks=chunks,
            output_text="",
            prompt_tokens_reported=prompt_tokens_reported,
            completion_tokens_reported=completion_tokens_reported,
            total_tokens_reported=total_tokens_reported,
        )
    except Exception as exc:  # noqa: BLE001
        return StreamResponse(
            success=False,
            http_status=http_status,
            error=str(exc),
            first_token_timeout_triggered=False,
            ttft_seconds=None,
            total_seconds=None,
            decode_seconds=None,
            raw_chunks=chunks,
            output_text="",
            prompt_tokens_reported=prompt_tokens_reported,
            completion_tokens_reported=completion_tokens_reported,
            total_tokens_reported=total_tokens_reported,
        )

    output_text = "".join(generated_parts)
    total_seconds = None if end_time is None else end_time - start
    ttft_seconds = None if first_token_time is None else first_token_time - start
    decode_seconds = None
    if first_token_time is not None and end_time is not None and end_time > first_token_time:
        decode_seconds = end_time - first_token_time

    if http_status is not None and 200 <= http_status < 300 and not output_text:
        return StreamResponse(
            success=False,
            http_status=http_status,
            error="stream completed without any output tokens",
            first_token_timeout_triggered=False,
            ttft_seconds=ttft_seconds,
            total_seconds=total_seconds,
            decode_seconds=decode_seconds,
            raw_chunks=chunks,
            output_text="",
            prompt_tokens_reported=prompt_tokens_reported,
            completion_tokens_reported=completion_tokens_reported,
            total_tokens_reported=total_tokens_reported,
        )

    return StreamResponse(
        success=True,
        http_status=http_status,
        error="",
        first_token_timeout_triggered=False,
        ttft_seconds=ttft_seconds,
        total_seconds=total_seconds,
        decode_seconds=decode_seconds,
        raw_chunks=chunks,
        output_text=output_text,
        prompt_tokens_reported=prompt_tokens_reported,
        completion_tokens_reported=completion_tokens_reported,
        total_tokens_reported=total_tokens_reported,
    )


def fetch_url(url: str, api_key: str, timeout: int) -> str:
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_prometheus_metrics(text: str) -> Dict[str, List[float]]:
    metrics: Dict[str, List[float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if " " not in line:
            continue
        left, right = line.rsplit(" ", 1)
        name = left.split("{", 1)[0]
        try:
            value = float(right)
        except ValueError:
            continue
        metrics.setdefault(name, []).append(value)
    return metrics


def collect_metrics_snapshot(base_url: str, api_key: str, timeout: int) -> MetricsSnapshot:
    notes: List[str] = []
    kv_cache_usage = None
    prefix_queries = None
    prefix_hits = None

    normalized_base = base_url.rstrip("/")
    if normalized_base.endswith("/v1"):
        metrics_root = normalized_base[:-3]
    else:
        metrics_root = normalized_base
    metrics_url = f"{metrics_root}/metrics"
    try:
        text = fetch_url(metrics_url, api_key, timeout=min(timeout, 30))
        metrics = parse_prometheus_metrics(text)
        kv_cache_usage = find_metric_value(metrics, ["kv_cache_usage_perc", "gpu_cache_usage_perc"])
        prefix_queries = find_metric_value(metrics, ["prefix_cache_queries", "prefix_cache_query_total", "prefix_cache_requests"])
        prefix_hits = find_metric_value(metrics, ["prefix_cache_hits", "prefix_cache_hit_total"])
    except Exception as exc:  # noqa: BLE001
        notes.append(f"metrics_unavailable: {exc}")

    gpu_util, gpu_mem_mb, gpu_mem_util, gpu_note = collect_nvidia_smi_snapshot()
    if gpu_note:
        notes.append(gpu_note)

    return MetricsSnapshot(
        collected=kv_cache_usage is not None or prefix_queries is not None or prefix_hits is not None or gpu_util is not None,
        source="vllm_metrics+nvidia_smi",
        kv_cache_usage_perc=kv_cache_usage,
        prefix_cache_queries=prefix_queries,
        prefix_cache_hits=prefix_hits,
        gpu_utilization_avg=gpu_util,
        gpu_memory_used_mb_total=gpu_mem_mb,
        gpu_memory_utilization_avg=gpu_mem_util,
        raw_notes=notes,
    )


def collect_nvidia_smi_snapshot() -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None, None, None, "nvidia-smi_not_found"
    except subprocess.CalledProcessError as exc:
        return None, None, None, f"nvidia_smi_failed: {exc}"

    utils = []
    used = []
    mem_utils = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpu_util = float(parts[0])
            mem_used = float(parts[1])
            mem_total = float(parts[2])
        except ValueError:
            continue
        utils.append(gpu_util)
        used.append(mem_used)
        if mem_total > 0:
            mem_utils.append(mem_used / mem_total * 100.0)

    if not utils:
        return None, None, None, "nvidia_smi_no_rows"

    avg_gpu_util = sum(utils) / len(utils)
    total_mem_used = sum(used)
    avg_mem_util = sum(mem_utils) / len(mem_utils) if mem_utils else None
    return avg_gpu_util, total_mem_used, avg_mem_util, None


def run_command_text(cmd: List[str]) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return True, result.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def read_os_release() -> str:
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            data = f.read()
    except Exception:
        return platform_fallback()

    fields = {}
    for line in data.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip().strip('"')
    return fields.get("PRETTY_NAME") or fields.get("NAME") or platform_fallback()


def platform_fallback() -> str:
    ok, text = run_command_text(["uname", "-s"])
    return text if ok else "unknown"


def collect_environment_snapshot() -> EnvironmentSnapshot:
    notes: List[str] = []

    hostname = socket.gethostname()
    platform_name = sys.platform

    ok, kernel = run_command_text(["uname", "-a"])
    if not ok:
        notes.append(f"uname_failed: {kernel}")
        kernel = "unknown"

    os_release = read_os_release()

    cpu_model = "unknown"
    cpu_logical = os.cpu_count()
    cpu_physical = None
    ok, lscpu_text = run_command_text(["lscpu"])
    if ok:
        for line in lscpu_text.splitlines():
            if ":" not in line:
                continue
            key, value = [item.strip() for item in line.split(":", 1)]
            if key == "Model name" and value:
                cpu_model = value
            elif key == "CPU(s)" and value.isdigit():
                cpu_logical = int(value)
            elif key == "Core(s) per socket":
                try:
                    cores_per_socket = int(value)
                except ValueError:
                    cores_per_socket = None
            elif key == "Socket(s)":
                try:
                    sockets = int(value)
                except ValueError:
                    sockets = None
        if "cores_per_socket" in locals() and "sockets" in locals():
            if cores_per_socket is not None and sockets is not None:
                cpu_physical = cores_per_socket * sockets
    else:
        notes.append(f"lscpu_failed: {lscpu_text}")

    total_memory_gb = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        total_memory_gb = page_size * phys_pages / (1024 ** 3)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"memory_detect_failed: {exc}")

    disk_root_total_gb = None
    disk_root_free_gb = None
    try:
        stat = os.statvfs("/")
        disk_root_total_gb = stat.f_blocks * stat.f_frsize / (1024 ** 3)
        disk_root_free_gb = stat.f_bavail * stat.f_frsize / (1024 ** 3)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"disk_detect_failed: {exc}")

    gpu_summary: List[str] = []
    nvidia_summary = ""
    ok, gpu_list_text = run_command_text(["nvidia-smi", "-L"])
    if ok:
        gpu_summary = [line.strip() for line in gpu_list_text.splitlines() if line.strip()]
    else:
        notes.append(f"nvidia_smi_list_failed: {gpu_list_text}")

    ok, gpu_query_text = run_command_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if ok:
        nvidia_summary = "; ".join(line.strip() for line in gpu_query_text.splitlines() if line.strip())
    else:
        notes.append(f"nvidia_smi_query_failed: {gpu_query_text}")
        nvidia_summary = gpu_query_text

    return EnvironmentSnapshot(
        hostname=hostname,
        platform=platform_name,
        kernel=kernel,
        os_release=os_release,
        python_version=sys.version.split()[0],
        cpu_model=cpu_model,
        cpu_cores_logical=cpu_logical,
        cpu_cores_physical=cpu_physical,
        total_memory_gb=total_memory_gb,
        disk_root_total_gb=disk_root_total_gb,
        disk_root_free_gb=disk_root_free_gb,
        gpu_summary=gpu_summary,
        nvidia_smi_summary=nvidia_summary,
        raw_notes=notes,
    )


def find_metric_value(metrics: Dict[str, List[float]], patterns: List[str]) -> Optional[float]:
    for name, values in metrics.items():
        for pattern in patterns:
            if pattern in name:
                return sum(values)
    return None


def load_testset(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bench_cases_from_testset(testset: dict, tokenizer_ref: str, estimator: TokenEstimator) -> List[dict]:
    trace = testset["agent_trace_case"]
    base_messages = trace["messages"]
    final_prompt = trace.get("final_user_prompts", ["请继续。"])[0]
    amplify_cfg = trace.get("amplify") or {}

    cases = []
    for milestone in trace["milestones"]:
        tool_calls_included = int(milestone["tool_calls_included"])
        messages = build_cumulative_agent_messages(
            base_messages,
            tool_calls_included,
            final_prompt,
            amplify_cfg,
        )
        prompt_tokens_estimated = estimator.count_messages(tokenizer_ref, messages)
        cases.append(
            {
                "name": milestone["name"],
                "description": milestone.get("goal", ""),
                "messages": messages,
                "prompt_tokens_estimated": prompt_tokens_estimated,
                "tool_calls_included": tool_calls_included,
            }
        )

    return cases


def build_cumulative_agent_messages(
    messages: List[dict],
    tool_calls_included: int,
    final_prompt: str,
    amplify_cfg: dict,
) -> List[dict]:
    selected: List[dict] = []
    seen_tool_results = 0

    for message in messages:
        normalized = maybe_amplify_message(message, amplify_cfg)
        selected.append(normalized)
        if message.get("role") == "tool":
            seen_tool_results += 1
            if seen_tool_results >= tool_calls_included:
                break

    if seen_tool_results < tool_calls_included:
        raise ValueError(
            f"Requested tool_calls_included={tool_calls_included}, but only found {seen_tool_results} tool results."
        )

    return render_agent_trace_as_openai_messages(selected, final_prompt)


def maybe_amplify_message(message: dict, amplify_cfg: dict) -> dict:
    if not amplify_cfg.get("enabled"):
        return message
    if message.get("role") != "tool":
        return message

    repeat_times = int(amplify_cfg.get("repeat_each_tool_result", 1))
    if repeat_times <= 1:
        return message
    separator = str(amplify_cfg.get("separator", "\n\n"))
    content = str(message.get("content", ""))
    if not content:
        return message

    expanded_content = separator.join(content for _ in range(repeat_times))
    amplified = dict(message)
    amplified["content"] = expanded_content
    return amplified


def render_agent_trace_as_openai_messages(messages: List[dict], final_prompt: str) -> List[dict]:
    system_content = "你是一个负责线上稳定性的运维代理。请基于累计历史继续分析。"
    transcript_parts: List[str] = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            content = str(message.get("content", "")).strip()
            if content:
                system_content = content
            continue

        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            content = str(message.get("content", "")).strip()
            if tool_calls:
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function") or {}
                    name = function.get("name", "")
                    arguments = function.get("arguments", "")
                    call_id = tool_call.get("id", "")
                    transcript_parts.append(
                        f"[assistant_tool_call:{call_id}]\nfunction={name}\narguments={arguments}"
                    )
            elif content:
                transcript_parts.append(f"[assistant]\n{content}")
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id", "")
            content = str(message.get("content", "")).strip()
            transcript_parts.append(f"[tool_result:{tool_call_id}]\n{content}")
            continue

        content = str(message.get("content", "")).strip()
        if content:
            transcript_parts.append(f"[{role}]\n{content}")

    user_content = (
        "以下是到当前步骤为止的累计 agent 执行历史。"
        "这些历史已经按时间顺序展开，包含用户任务、assistant 的工具调用决定，以及各工具返回。"
        "请基于全部历史继续处理，不要忽略前文。\n\n"
        + "\n\n".join(transcript_parts)
        + "\n\n[final_request]\n"
        + final_prompt
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def parse_context_sizes(raw: str) -> List[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def progress_reporter(stop_event: threading.Event, start_time: float, interval: int) -> None:
    while not stop_event.wait(interval):
        elapsed = time.perf_counter() - start_time
        print(f"  waiting... elapsed={elapsed:.1f}s", flush=True)


def run_with_progress(
    runner: Callable[[], StreamResponse],
    progress_interval: int,
) -> StreamResponse:
    run_start = time.perf_counter()
    stop_event = threading.Event()
    reporter = threading.Thread(
        target=progress_reporter,
        args=(stop_event, run_start, progress_interval),
        daemon=True,
    )
    reporter.start()
    try:
        return runner()
    finally:
        stop_event.set()
        reporter.join(timeout=0.1)


def evaluate_smoke_case(case: dict, output_text: str) -> Tuple[bool, str]:
    text = output_text.strip()
    regex = case.get("regex")
    if regex is not None:
        if re.fullmatch(regex, text):
            return True, "matched regex rule"
        return False, f"output does not match regex '{regex}': '{text[:80]}'"

    exact = case.get("exact")
    if exact is not None:
        if text == exact:
            return True, "matched exact output"
        return False, f"expected exact '{exact}', got '{text[:80]}'"

    expected_contains = case.get("contains") or []
    missing = [item for item in expected_contains if item not in text]
    if missing:
        return False, f"missing expected fragments: {missing}"

    if not text:
        return False, "empty output"

    return True, "basic content checks passed"


def make_output_tps_estimated(completion_tokens: int, decode_seconds: Optional[float]) -> Optional[float]:
    if completion_tokens <= 0 or decode_seconds is None or decode_seconds <= 0:
        return None
    return completion_tokens / decode_seconds


def build_output_preview(text: str, limit: int = 120) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def run_smoke_tests(
    base_url: str,
    api_key: str,
    models: List[dict],
    temperature: float,
    timeout: int,
    first_token_timeout: int,
    progress_interval: int,
    estimator: TokenEstimator,
    thinking_mode: str,
) -> List[SmokeResult]:
    results: List[SmokeResult] = []
    suite_start = time.perf_counter()

    for model_cfg in models:
        model_name = model_cfg["name"]
        tokenizer_ref = model_cfg.get("tokenizer", model_name)
        model_start = time.perf_counter()
        print(f"\n[MODEL][SMOKE] {model_name}", flush=True)
        print(f"  tokenizer: {estimator.status(tokenizer_ref)}", flush=True)

        for case in SMOKE_CASES:
            request_messages = apply_thinking_mode(case["messages"], thinking_mode)
            print(f"[RUN][SMOKE] {case['name']} - {case['description']}", flush=True)
            metrics_before = collect_metrics_snapshot(base_url, api_key, timeout)
            prompt_tokens_estimated = estimator.count_messages(tokenizer_ref, request_messages)
            print(
                f"  request: prompt_tokens_estimated={prompt_tokens_estimated} "
                f"max_tokens={int(case['max_tokens'])} temperature={temperature} timeout={timeout}s "
                f"thinking_mode={thinking_mode}",
                flush=True,
            )
            print(f"  metrics_before: {format_metrics_snapshot(metrics_before)}", flush=True)

            response = run_with_progress(
                lambda: run_stream_chat(
                    base_url=base_url,
                    model=model_name,
                    api_key=api_key,
                    messages=request_messages,
                    max_tokens=int(case["max_tokens"]),
                    temperature=temperature,
                    timeout=timeout,
                    first_token_timeout=first_token_timeout,
                    on_first_token=lambda elapsed: print(
                        f"  first token arrived at {elapsed:.3f}s",
                        flush=True,
                    ),
                ),
                progress_interval=progress_interval,
            )
            metrics_after = collect_metrics_snapshot(base_url, api_key, timeout)
            completion_tokens_estimated = estimator.count_text(tokenizer_ref, response.output_text)
            output_tps = make_output_tps_estimated(completion_tokens_estimated, response.decode_seconds)
            check_passed, check_message = evaluate_smoke_case(case, response.output_text)
            success = response.success and check_passed

            result = SmokeResult(
                model=model_name,
                case_name=case["name"],
                case_description=case["description"],
                success=success,
                http_status=response.http_status,
                error=response.error,
                ttft_seconds=response.ttft_seconds,
                total_seconds=response.total_seconds,
                decode_seconds=response.decode_seconds,
                raw_chunks=response.raw_chunks,
                prompt_tokens_estimated=prompt_tokens_estimated,
                completion_tokens_estimated=completion_tokens_estimated,
                prompt_tokens_reported=response.prompt_tokens_reported,
                completion_tokens_reported=response.completion_tokens_reported,
                total_tokens_reported=response.total_tokens_reported,
                output_tps_estimated=output_tps,
                output_text=response.output_text,
                output_preview=build_output_preview(response.output_text),
                check_passed=check_passed,
                check_message=check_message,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
            )
            results.append(result)

            if result.success:
                print(
                    f"  ok ttft={fmt(result.ttft_seconds)}s total={fmt(result.total_seconds)}s "
                    f"completion_tokens={result.completion_tokens_estimated} tps={fmt(result.output_tps_estimated)} "
                    f"check={result.check_message}",
                    flush=True,
                )
            else:
                print(
                    f"  fail status={result.http_status} error={shorten(result.error, 160)} "
                    f"check={result.check_message}",
                    flush=True,
                )
            print(f"  output_preview: {result.output_preview}", flush=True)
            print(f"  metrics_after: {format_metrics_snapshot(metrics_after)}", flush=True)
            if response.first_token_timeout_triggered:
                print(
                    f"  stop: first token exceeded {first_token_timeout}s, aborting remaining smoke tests",
                    flush=True,
                )
                return results

        model_results = [item for item in results if item.model == model_name]
        ok_count = sum(1 for item in model_results if item.success)
        print(
            f"[MODEL][SMOKE][DONE] {model_name} success={ok_count}/{len(model_results)} "
            f"elapsed={fmt(time.perf_counter() - model_start)}s",
            flush=True,
        )

    print(f"[SUITE][SMOKE][DONE] elapsed={fmt(time.perf_counter() - suite_start)}s", flush=True)

    return results


def run_bench_tests(
    base_url: str,
    api_key: str,
    models: List[dict],
    testset_path: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    first_token_timeout: int,
    progress_interval: int,
    estimator: TokenEstimator,
    thinking_mode: str,
) -> List[BenchResult]:
    results: List[BenchResult] = []
    suite_start = time.perf_counter()
    testset = load_testset(testset_path)
    bench_thinking_modes = ["think", "no_think"]

    for model_cfg in models:
        model_name = model_cfg["name"]
        tokenizer_ref = model_cfg.get("tokenizer", model_name)
        model_start = time.perf_counter()
        print(f"\n[MODEL][BENCH] {model_name}", flush=True)
        print(f"  tokenizer: {estimator.status(tokenizer_ref)}", flush=True)
        bench_cases = build_bench_cases_from_testset(testset, tokenizer_ref, estimator)
        mode_aborted = {mode: False for mode in bench_thinking_modes}
        previous_prompt_tokens = {mode: -1 for mode in bench_thinking_modes}
        for case in bench_cases:
            for bench_mode in bench_thinking_modes:
                if mode_aborted[bench_mode]:
                    continue
                request_messages = apply_thinking_mode(case["messages"], bench_mode)
                prompt_tokens_estimated = estimator.count_messages(tokenizer_ref, request_messages)
                if prompt_tokens_estimated <= previous_prompt_tokens[bench_mode]:
                    raise ValueError(
                        f"Bench case '{case['name']}' prompt length did not grow monotonically "
                        f"for mode '{bench_mode}': {prompt_tokens_estimated} <= {previous_prompt_tokens[bench_mode]}"
                    )
                previous_prompt_tokens[bench_mode] = prompt_tokens_estimated
                print(
                    f"[RUN][BENCH] case={case['name']} mode={bench_mode} "
                    f"tool_calls_included={case['tool_calls_included']} "
                    f"prompt_tokens_estimated={prompt_tokens_estimated}",
                    flush=True,
                )
                metrics_before = collect_metrics_snapshot(base_url, api_key, timeout)
                print(
                    f"  request: prompt_tokens_estimated={prompt_tokens_estimated} "
                    f"max_tokens={max_tokens} temperature={temperature} timeout={timeout}s "
                    f"thinking_mode={bench_mode}",
                    flush=True,
                )
                print(f"  metrics_before: {format_metrics_snapshot(metrics_before)}", flush=True)
                response = run_with_progress(
                    lambda: run_stream_chat(
                        base_url=base_url,
                        model=model_name,
                        api_key=api_key,
                        messages=request_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout,
                        first_token_timeout=first_token_timeout,
                        on_first_token=lambda elapsed: print(
                            f"  first token arrived at {elapsed:.3f}s",
                            flush=True,
                        ),
                    ),
                    progress_interval=progress_interval,
                )
                metrics_after = collect_metrics_snapshot(base_url, api_key, timeout)
                completion_tokens_estimated = estimator.count_text(tokenizer_ref, response.output_text)
                output_tps = make_output_tps_estimated(completion_tokens_estimated, response.decode_seconds)

                result = BenchResult(
                    model=model_name,
                    case_name=case["name"],
                    thinking_mode=bench_mode,
                    context_tokens_target=prompt_tokens_estimated,
                    success=response.success,
                    http_status=response.http_status,
                    error=response.error,
                    ttft_seconds=response.ttft_seconds,
                    total_seconds=response.total_seconds,
                    decode_seconds=response.decode_seconds,
                    raw_chunks=response.raw_chunks,
                    prompt_tokens_estimated=prompt_tokens_estimated,
                    completion_tokens_estimated=completion_tokens_estimated,
                    prompt_tokens_reported=response.prompt_tokens_reported,
                    completion_tokens_reported=response.completion_tokens_reported,
                    total_tokens_reported=response.total_tokens_reported,
                    output_tps_estimated=output_tps,
                    output_preview=build_output_preview(response.output_text),
                    metrics_before=metrics_before,
                    metrics_after=metrics_after,
                )
                results.append(result)

                if result.success:
                    print(
                        f"  ok ttft={fmt(result.ttft_seconds)}s total={fmt(result.total_seconds)}s "
                        f"completion_tokens={result.completion_tokens_estimated} tps={fmt(result.output_tps_estimated)}",
                        flush=True,
                    )
                else:
                    print(
                        f"  fail status={result.http_status} error={shorten(result.error, 160)}",
                        flush=True,
                    )
                print(f"  output_preview: {result.output_preview}", flush=True)
                print(f"  metrics_after: {format_metrics_snapshot(metrics_after)}", flush=True)
                if response.first_token_timeout_triggered:
                    mode_aborted[bench_mode] = True
                    print(
                        f"  stop: first token exceeded {first_token_timeout}s, "
                        f"aborting remaining bench tests for mode={bench_mode}",
                        flush=True,
                    )

        model_results = [item for item in results if item.model == model_name]
        ok_count = sum(1 for item in model_results if item.success)
        print(
            f"[MODEL][BENCH][DONE] {model_name} success={ok_count}/{len(model_results)} "
            f"elapsed={fmt(time.perf_counter() - model_start)}s",
            flush=True,
        )

    print(f"[SUITE][BENCH][DONE] elapsed={fmt(time.perf_counter() - suite_start)}s", flush=True)

    return results


def fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def average(values: List[Optional[float]]) -> Optional[float]:
    normalized = [value for value in values if value is not None]
    if not normalized:
        return None
    return sum(normalized) / len(normalized)


def maximum(values: List[Optional[float]]) -> Optional[float]:
    normalized = [value for value in values if value is not None]
    if not normalized:
        return None
    return max(normalized)


def shorten(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def render_metrics_delta(before: MetricsSnapshot, after: MetricsSnapshot) -> str:
    parts = []

    if after.kv_cache_usage_perc is not None:
        kv_text = f"kv_cache={after.kv_cache_usage_perc:.2f}%"
        if before.kv_cache_usage_perc is not None:
            kv_text += f" (before={before.kv_cache_usage_perc:.2f}%)"
        parts.append(kv_text)

    hit_rate = None
    if (
        before.prefix_cache_queries is not None
        and before.prefix_cache_hits is not None
        and after.prefix_cache_queries is not None
        and after.prefix_cache_hits is not None
    ):
        query_delta = after.prefix_cache_queries - before.prefix_cache_queries
        hit_delta = after.prefix_cache_hits - before.prefix_cache_hits
        if query_delta > 0:
            hit_rate = hit_delta / query_delta * 100.0
            parts.append(f"prefix_cache_hit_rate={hit_rate:.2f}%")

    if after.gpu_utilization_avg is not None:
        parts.append(f"gpu_util_avg={after.gpu_utilization_avg:.2f}%")
    if after.gpu_memory_utilization_avg is not None:
        parts.append(f"gpu_mem_util_avg={after.gpu_memory_utilization_avg:.2f}%")
    if after.gpu_memory_used_mb_total is not None:
        parts.append(f"gpu_mem_used_total_mb={after.gpu_memory_used_mb_total:.0f}")

    if after.raw_notes:
        parts.append("notes=" + "; ".join(after.raw_notes))

    return "<br>".join(parts) if parts else ""


def format_metrics_snapshot(snapshot: MetricsSnapshot) -> str:
    parts = []
    if snapshot.kv_cache_usage_perc is not None:
        parts.append(f"kv_cache={snapshot.kv_cache_usage_perc:.2f}%")
    if snapshot.prefix_cache_queries is not None:
        parts.append(f"prefix_queries={snapshot.prefix_cache_queries:.0f}")
    if snapshot.prefix_cache_hits is not None:
        parts.append(f"prefix_hits={snapshot.prefix_cache_hits:.0f}")
    if snapshot.gpu_utilization_avg is not None:
        parts.append(f"gpu_util={snapshot.gpu_utilization_avg:.2f}%")
    if snapshot.gpu_memory_utilization_avg is not None:
        parts.append(f"gpu_mem_util={snapshot.gpu_memory_utilization_avg:.2f}%")
    if snapshot.gpu_memory_used_mb_total is not None:
        parts.append(f"gpu_mem_mb={snapshot.gpu_memory_used_mb_total:.0f}")
    if snapshot.raw_notes:
        parts.append("notes=" + "; ".join(snapshot.raw_notes))
    return ", ".join(parts) if parts else "no metrics collected"


def render_markdown_report(
    base_url: str,
    mode: str,
    models: List[dict],
    context_sizes: List[int],
    testset_path: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
    first_token_timeout: int,
    thinking_mode: str,
    environment: EnvironmentSnapshot,
    estimator: TokenEstimator,
    smoke_results: List[SmokeResult],
    bench_results: List[BenchResult],
) -> str:
    def get_bench_row(model_name: str, case_name: str, mode_name: str) -> Optional[BenchResult]:
        for row in bench_results:
            if row.model == model_name and row.case_name == case_name and row.thinking_mode == mode_name:
                return row
        return None

    lines: List[str] = []
    lines.append("# Model Service Benchmark Report")
    lines.append("")
    lines.append("## Test Config")
    lines.append("")
    lines.append(f"- Script version: `{SCRIPT_VERSION}`")
    lines.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Base URL: `{base_url}`")
    lines.append(f"- Mode: `{mode}`")
    lines.append(f"- Models: `{', '.join(item['name'] for item in models)}`")
    lines.append(f"- Context sizes: `{','.join(str(x) for x in context_sizes)}`")
    lines.append(f"- Bench testset: `{testset_path}`")
    lines.append(f"- Thinking mode: `{thinking_mode}`")
    lines.append(f"- Max output tokens: `{max_tokens}`")
    lines.append(f"- Temperature: `{temperature}`")
    lines.append(f"- Timeout per request: `{timeout}s`")
    lines.append(f"- First token timeout: `{first_token_timeout}s`")
    lines.append("")
    lines.append("## Tokenizer Status")
    lines.append("")
    for item in models:
        tokenizer_ref = item.get("tokenizer", item["name"])
        lines.append(f"- `{item['name']}` -> `{estimator.status(tokenizer_ref)}`")

    lines.append("")
    lines.append("## Environment Baseline")
    lines.append("")
    lines.append(f"- Hostname: `{environment.hostname}`")
    lines.append(f"- Platform: `{environment.platform}`")
    lines.append(f"- OS: `{environment.os_release}`")
    lines.append(f"- Kernel: `{environment.kernel}`")
    lines.append(f"- Python: `{environment.python_version}`")
    lines.append(f"- CPU model: `{environment.cpu_model}`")
    lines.append(f"- CPU logical cores: `{environment.cpu_cores_logical}`")
    lines.append(f"- CPU physical cores: `{environment.cpu_cores_physical}`")
    lines.append(f"- Total memory (GB): `{fmt(environment.total_memory_gb)}`")
    lines.append(f"- Root disk total/free (GB): `{fmt(environment.disk_root_total_gb)}` / `{fmt(environment.disk_root_free_gb)}`")
    lines.append(f"- GPU summary: `{'; '.join(environment.gpu_summary) if environment.gpu_summary else 'unavailable'}`")
    lines.append(f"- nvidia-smi summary: `{environment.nvidia_smi_summary or 'unavailable'}`")
    if environment.raw_notes:
        lines.append(f"- Notes: `{'; '.join(environment.raw_notes)}`")

    lines.append("")
    lines.append("## Model Summary")
    lines.append("")
    lines.append(
        "| Model | Smoke Success | Bench Success | Avg TTFT (s) | Max TTFT (s) | Avg Output TPS | Avg GPU Util (%) | Avg KV Cache (%) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in models:
        model_name = item["name"]
        model_smoke = [row for row in smoke_results if row.model == model_name]
        model_bench = [row for row in bench_results if row.model == model_name]
        all_rows = model_smoke + model_bench
        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    f"{sum(1 for row in model_smoke if row.success)}/{len(model_smoke)}" if model_smoke else "-",
                    f"{sum(1 for row in model_bench if row.success)}/{len(model_bench)}" if model_bench else "-",
                    fmt(average([row.ttft_seconds for row in all_rows])),
                    fmt(maximum([row.ttft_seconds for row in all_rows])),
                    fmt(average([row.output_tps_estimated for row in all_rows])),
                    fmt(average([row.metrics_after.gpu_utilization_avg for row in all_rows])),
                    fmt(average([row.metrics_after.kv_cache_usage_perc for row in all_rows])),
                ]
            )
            + " |"
        )

    if smoke_results:
        lines.append("")
        lines.append("## Smoke Results")
        lines.append("")
        lines.append(
            "| Model | Case | Success | HTTP | TTFT (s) | Total (s) | Prompt Tokens | Completion Tokens | Output TPS | Chunks | Check | Output Preview | Service Metrics |"
        )
        lines.append(
            "| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"
        )
        for row in smoke_results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.model,
                        row.case_name,
                        "Y" if row.success else "N",
                        str(row.http_status or ""),
                        fmt(row.ttft_seconds),
                        fmt(row.total_seconds),
                        str(row.prompt_tokens_reported or row.prompt_tokens_estimated),
                        str(row.completion_tokens_reported or row.completion_tokens_estimated),
                        fmt(row.output_tps_estimated),
                        str(row.raw_chunks),
                        row.check_message.replace("|", "\\|"),
                        row.output_preview.replace("|", "\\|"),
                        render_metrics_delta(row.metrics_before, row.metrics_after).replace("|", "\\|"),
                    ]
                )
                + " |"
            )

        smoke_failures = [item for item in smoke_results if not item.success]
        if smoke_failures:
            lines.append("")
            lines.append("### Smoke Failures")
            lines.append("")
            for item in smoke_failures:
                lines.append(f"#### {item.model} / {item.case_name}")
                lines.append("")
                lines.append(f"- HTTP status: `{item.http_status}`")
                lines.append(f"- Error: `{shorten(item.error, 500)}`")
                lines.append(f"- Check: `{item.check_message}`")
                lines.append(f"- Output: `{shorten(item.output_text, 300)}`")
                lines.append("")

    if bench_results:
        lines.append("")
        lines.append("## Bench Results")
        lines.append("")
        lines.append(
            "| Model | Case | Prompt Tokens (think/no_think) | Think OK | NoThink OK | Think TTFT (s) | NoThink TTFT (s) | Think Total (s) | NoThink Total (s) | Think Completion | NoThink Completion | Think Output | NoThink Output |"
        )
        lines.append(
            "| --- | --- | --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
        )
        comparison_keys = []
        seen = set()
        for row in bench_results:
            key = (row.model, row.case_name)
            if key not in seen:
                seen.add(key)
                comparison_keys.append(key)
        for model_name, case_name in comparison_keys:
            think_row = get_bench_row(model_name, case_name, "think")
            no_think_row = get_bench_row(model_name, case_name, "no_think")
            prompt_pair = f"{think_row.prompt_tokens_reported or think_row.prompt_tokens_estimated if think_row else ''}/{no_think_row.prompt_tokens_reported or no_think_row.prompt_tokens_estimated if no_think_row else ''}"
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_name,
                        case_name,
                        prompt_pair,
                        "Y" if think_row and think_row.success else "N",
                        "Y" if no_think_row and no_think_row.success else "N",
                        fmt(think_row.ttft_seconds) if think_row else "",
                        fmt(no_think_row.ttft_seconds) if no_think_row else "",
                        fmt(think_row.total_seconds) if think_row else "",
                        fmt(no_think_row.total_seconds) if no_think_row else "",
                        str(think_row.completion_tokens_reported or think_row.completion_tokens_estimated) if think_row else "",
                        str(no_think_row.completion_tokens_reported or no_think_row.completion_tokens_estimated) if no_think_row else "",
                        (think_row.output_preview if think_row else "").replace("|", "\\|"),
                        (no_think_row.output_preview if no_think_row else "").replace("|", "\\|"),
                    ]
                )
                + " |"
            )

        lines.append("")
        lines.append("### Bench Per-Mode Details")
        lines.append("")
        lines.append(
            "| Model | Case | Mode | Prompt Tokens | Success | HTTP | TTFT (s) | Total (s) | Decode (s) | Completion Tokens | Output TPS | Chunks | Output Preview | Service Metrics |"
        )
        lines.append(
            "| --- | --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
        )
        for row in bench_results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.model,
                        row.case_name,
                        row.thinking_mode,
                        str(row.prompt_tokens_reported or row.prompt_tokens_estimated),
                        "Y" if row.success else "N",
                        str(row.http_status or ""),
                        fmt(row.ttft_seconds),
                        fmt(row.total_seconds),
                        fmt(row.decode_seconds),
                        str(row.completion_tokens_reported or row.completion_tokens_estimated),
                        fmt(row.output_tps_estimated),
                        str(row.raw_chunks),
                        row.output_preview.replace("|", "\\|"),
                        render_metrics_delta(row.metrics_before, row.metrics_after).replace("|", "\\|"),
                    ]
                )
                + " |"
            )

        bench_failures = [item for item in bench_results if not item.success]
        if bench_failures:
            lines.append("")
            lines.append("### Bench Failures")
            lines.append("")
            for item in bench_failures:
                lines.append(f"#### {item.model} / {item.case_name}")
                lines.append("")
                lines.append(f"- HTTP status: `{item.http_status}`")
                lines.append(f"- Error: `{shorten(item.error, 500)}`")
                lines.append("")

    lines.append("")
    lines.append("## Raw Results")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "base_url": base_url,
                "mode": mode,
                "models": models,
                "context_sizes": context_sizes,
                "testset_path": testset_path,
                "thinking_mode": thinking_mode,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout": timeout,
                "first_token_timeout": first_token_timeout,
                "environment": asdict(environment),
                "smoke_results": [asdict(item) for item in smoke_results],
                "bench_results": [asdict(item) for item in bench_results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def print_smoke_summary(results: List[SmokeResult]) -> None:
    if not results:
        return
    print("\n[SUMMARY][SMOKE]", flush=True)
    print("model\tcase\tok\tstatus\tttft_s\ttotal_s\tprompt_toks\tcompletion_toks\tout_tps\tchunks", flush=True)
    for row in results:
        print(
            "\t".join(
                [
                    row.model,
                    row.case_name,
                    "Y" if row.success else "N",
                    str(row.http_status or ""),
                    fmt(row.ttft_seconds),
                    fmt(row.total_seconds),
                    str(row.prompt_tokens_reported or row.prompt_tokens_estimated),
                    str(row.completion_tokens_reported or row.completion_tokens_estimated),
                    fmt(row.output_tps_estimated),
                    str(row.raw_chunks),
                ]
            ),
            flush=True,
        )


def print_bench_summary(results: List[BenchResult]) -> None:
    if not results:
        return
    print("\n[SUMMARY][BENCH]", flush=True)
    print("model\tcase\tmode\tprompt_toks\tok\tstatus\tttft_s\ttotal_s\tdecode_s\tcompletion_toks\tout_tps\tchunks", flush=True)
    for row in results:
        print(
            "\t".join(
                [
                    row.model,
                    row.case_name,
                    row.thinking_mode,
                    str(row.context_tokens_target),
                    "Y" if row.success else "N",
                    str(row.http_status or ""),
                    fmt(row.ttft_seconds),
                    fmt(row.total_seconds),
                    fmt(row.decode_seconds),
                    str(row.completion_tokens_reported or row.completion_tokens_estimated),
                    fmt(row.output_tps_estimated),
                    str(row.raw_chunks),
                ]
            ),
            flush=True,
        )


def _safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def default_markdown_output(models: List[dict]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_part = "_".join(item["name"] for item in models)
    model_part = re.sub(r"[^A-Za-z0-9._-]+", "_", model_part).strip("_")
    if not model_part:
        model_part = "models"
    return f"{DEFAULT_REPORT_PREFIX}_{model_part}_{ts}.md"


def parse_models_arg(raw: str) -> List[dict]:
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    if not requested:
        return MODELS

    model_map = {item["name"]: item for item in MODELS}
    selected = []
    for name in requested:
        if name in model_map:
            selected.append(model_map[name])
        else:
            selected.append({"name": name, "tokenizer": name})
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple OpenAI-compatible model services with streaming Chinese test cases."
    )
    parser.add_argument("--base-url", default=BASE_URL, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=API_KEY, help="Bearer token")
    parser.add_argument("--models", default=",".join(item["name"] for item in MODELS), help="Comma-separated model names")
    parser.add_argument("--mode", choices=["smoke", "bench", "all"], default=DEFAULT_MODE, help="Run mode")
    parser.add_argument(
        "--context-sizes",
        default=",".join(str(x) for x in DEFAULT_CONTEXT_SIZES),
        help="Comma-separated prompt token targets, e.g. 128,512,2048",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max output tokens for bench mode")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Per request timeout in seconds")
    parser.add_argument("--first-token-timeout", type=int, default=DEFAULT_FIRST_TOKEN_TIMEOUT, help="Fail if first token is not received within this many seconds; abort remaining tests")
    parser.add_argument("--progress-interval", type=int, default=DEFAULT_PROGRESS_INTERVAL, help="Progress log interval in seconds")
    parser.add_argument("--testset", default=DEFAULT_TESTSET_PATH, help="Bench-mode testset JSON path")
    parser.add_argument("--thinking-mode", choices=["think", "no_think"], default=DEFAULT_THINKING_MODE, help="Request-side Qwen thinking control")
    parser.add_argument("--markdown-output", default="", help="Optional Markdown report file path")
    args = parser.parse_args()

    context_sizes = parse_context_sizes(args.context_sizes)
    if not context_sizes:
        print("No valid context sizes provided.", file=sys.stderr)
        return 2

    models = parse_models_arg(args.models)
    estimator = TokenEstimator()
    environment = collect_environment_snapshot()

    smoke_results: List[SmokeResult] = []
    bench_results: List[BenchResult] = []

    print(f"Benchmark target: {args.base_url}", flush=True)
    print(f"Models: {[item['name'] for item in models]}", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Context sizes: {context_sizes}", flush=True)
    print(f"Bench testset: {args.testset}", flush=True)
    print(f"Thinking mode: {args.thinking_mode}", flush=True)
    print(f"Max output tokens: {args.max_tokens}", flush=True)
    print(f"First token timeout: {args.first_token_timeout}", flush=True)
    print(
        f"Environment: host={environment.hostname} os={environment.os_release} "
        f"cpu={environment.cpu_model} mem_gb={fmt(environment.total_memory_gb)} "
        f"gpu_count={len(environment.gpu_summary)}",
        flush=True,
    )

    if args.mode in {"smoke", "all"}:
        smoke_results = run_smoke_tests(
            base_url=args.base_url,
            api_key=args.api_key,
            models=models,
            temperature=args.temperature,
            timeout=args.timeout,
            first_token_timeout=args.first_token_timeout,
            progress_interval=args.progress_interval,
            estimator=estimator,
            thinking_mode=args.thinking_mode,
        )

    if args.mode in {"bench", "all"}:
        bench_results = run_bench_tests(
            base_url=args.base_url,
            api_key=args.api_key,
            models=models,
            testset_path=args.testset,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            first_token_timeout=args.first_token_timeout,
            progress_interval=args.progress_interval,
            estimator=estimator,
            thinking_mode=args.thinking_mode,
        )

    print_smoke_summary(smoke_results)
    print_bench_summary(bench_results)

    report_path = args.markdown_output or default_markdown_output(models)
    report = render_markdown_report(
        base_url=args.base_url,
        mode=args.mode,
        models=models,
        context_sizes=context_sizes,
        testset_path=args.testset,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
        first_token_timeout=args.first_token_timeout,
        thinking_mode=args.thinking_mode,
        environment=environment,
        estimator=estimator,
        smoke_results=smoke_results,
        bench_results=bench_results,
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved markdown report to: {report_path}", flush=True)

    all_ok = True
    if smoke_results:
        all_ok = all_ok and all(item.success for item in smoke_results)
    if bench_results:
        all_ok = all_ok and all(item.success for item in bench_results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
