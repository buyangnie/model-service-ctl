#!/usr/bin/env bash
set -euo pipefail

#######################################
# 通用 vLLM 容器管理脚本（单服务版）
# 当前默认环境：
#   - 脚本目录: /opsfactory/model_service
#   - 模型目录: /data/models/Qwen3-30B-A3B-Thinking-2507
#######################################

#######################################
# Script Meta
#######################################
SCRIPT_NAME="$(basename "$0")"
SCRIPT_VERSION="2.0"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

ROOT_LOG_DIR="${BASE_DIR}/logs"
RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${ROOT_LOG_DIR}/run_${RUN_TS}"
mkdir -p "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/script.log"
LATEST_LOG="${ROOT_LOG_DIR}/latest.log"
LATEST_RUN="${ROOT_LOG_DIR}/latest_run"

mkdir -p "${ROOT_LOG_DIR}"
ln -sfn "${LOG_DIR}" "${LATEST_RUN}" 2>/dev/null || true
ln -sfn "${RUN_LOG}" "${LATEST_LOG}" 2>/dev/null || true

#######################################
# Global Config
# 可通过环境变量覆盖
#######################################
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-x86_64}"
API_KEY="${API_KEY:-change-me}"

MODELS_ROOT="${MODELS_ROOT:-/data/models}"
HF_CACHE="${HF_CACHE:-/data/hf_cache}"

CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen3-30b-a3b-thinking-2507}"
MODEL_DIR="${MODEL_DIR:-/data/models/Qwen3-30B-A3B-Thinking-2507}"
MODEL_NAME="${MODEL_NAME:-Qwen3-30B-A3B-Thinking-2507}"

HOST_PORT="${HOST_PORT:-8000}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"

GPU_DEVICES="${GPU_DEVICES:-device=0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"

DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
# Official Qwen guidance for this model uses 262144 context on vLLM. Keep it
# as the default and let callers override if they want a lower startup target.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

# vLLM's official Qwen tool-calling guidance uses the Hermes-style parser.
# This model is thinking-only and the model card explicitly states that
# specifying enable_thinking is no longer required. The vLLM image available in
# this environment also rejects --enable-reasoning, so keep reasoning-related
# CLI flags disabled by default.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
REASONING_PARSER="${REASONING_PARSER:-}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
ENABLE_REASONING="${ENABLE_REASONING:-0}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

HEALTH_RETRIES="${HEALTH_RETRIES:-600}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-2}"
HEALTH_CONNECT_TIMEOUT="${HEALTH_CONNECT_TIMEOUT:-2}"
HEALTH_MAX_TIME="${HEALTH_MAX_TIME:-8}"

CURL_BIN="${CURL_BIN:-curl}"
DOCKER_BIN="${DOCKER_BIN:-docker}"

MODEL_DIR_REAL="${MODEL_DIR}"
MODEL_EXTRA_MOUNT=""

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

#######################################
# Logging Helpers
#######################################
ts() {
  date '+%F %T'
}

log() {
  echo "[$(ts)] INFO  $*" | tee -a "${RUN_LOG}"
}

warn() {
  echo "[$(ts)] WARN  $*" | tee -a "${RUN_LOG}" >&2
}

err() {
  echo "[$(ts)] ERROR $*" | tee -a "${RUN_LOG}" >&2
}

divider() {
  printf '%s\n' "================================================================" | tee -a "${RUN_LOG}"
}

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} validate
  ${SCRIPT_NAME} startup
  ${SCRIPT_NAME} stop
  ${SCRIPT_NAME} restart
  ${SCRIPT_NAME} status
  ${SCRIPT_NAME} logs
  ${SCRIPT_NAME} debug

Current defaults:
  MODEL_DIR=${MODEL_DIR}
  MODEL_NAME=${MODEL_NAME}
  HOST_PORT=${HOST_PORT}
  GPU_DEVICES=${GPU_DEVICES}
  TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
  MAX_MODEL_LEN=${MAX_MODEL_LEN}
  MAX_NUM_SEQS=${MAX_NUM_SEQS}
  TOOL_CALL_PARSER=${TOOL_CALL_PARSER}
  REASONING_PARSER=${REASONING_PARSER}
  ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE}
  ENABLE_REASONING=${ENABLE_REASONING}
  LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY}
  CHAT_TEMPLATE=${CHAT_TEMPLATE}
  TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE}
  DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE}
  ENFORCE_EAGER=${ENFORCE_EAGER}

Logs:
  root dir : ${ROOT_LOG_DIR}
  this run : ${LOG_DIR}
  latest   : ${LATEST_RUN}
EOF
}

#######################################
# Basic Helpers
#######################################
require_bin() {
  local bin="$1"
  command -v "$bin" >/dev/null 2>&1 || {
    err "Required command not found: $bin"
    exit 1
  }
}

container_exists() {
  ${DOCKER_BIN} ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

container_running() {
  ${DOCKER_BIN} ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

remove_container_if_exists() {
  if container_exists; then
    log "Removing existing container: ${CONTAINER_NAME}"
    ${DOCKER_BIN} rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  else
    log "Container not found, skip remove: ${CONTAINER_NAME}"
  fi
}

stop_container_if_exists() {
  if container_exists; then
    log "Stopping container: ${CONTAINER_NAME}"
    ${DOCKER_BIN} rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  else
    log "Container not found, skip stop: ${CONTAINER_NAME}"
  fi
}

port_check() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -lntp | awk -v p=":${port}" '$4 ~ p {print}'
  elif command -v netstat >/dev/null 2>&1; then
    netstat -lntp 2>/dev/null | awk -v p=":${port}" '$4 ~ p {print}'
  else
    echo "ss/netstat not found"
  fi
}

is_port_listening() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -lnt | awk '{print $4}' | grep -qE "[:.]${port}$"
  elif command -v netstat >/dev/null 2>&1; then
    netstat -lnt 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}$"
  else
    return 1
  fi
}

capture_cmd() {
  local outfile="$1"
  shift
  {
    echo "### CAPTURE_TIMESTAMP ###"
    echo "$(ts)"
    echo "### COMMAND ###"
    printf '%q ' "$@"
    echo
    echo "### OUTPUT_BEGIN ###"
    "$@"
    local rc=$?
    echo
    echo "### OUTPUT_END ###"
    echo "exit_code=${rc}"
  } >"${outfile}" 2>&1 || true
}

normalize_path() {
  local path="${1:-}"
  if [[ -z "${path}" ]]; then
    echo "${path}"
    return 0
  fi

  if [[ "${path}" == "/" ]]; then
    echo "/"
    return 0
  fi

  while [[ "${path}" == */ ]]; do
    path="${path%/}"
  done
  echo "${path}"
}

resolve_model_dir() {
  local resolved=""
  MODELS_ROOT="$(normalize_path "${MODELS_ROOT}")"
  HF_CACHE="$(normalize_path "${HF_CACHE}")"
  MODEL_DIR="$(normalize_path "${MODEL_DIR}")"

  if command -v readlink >/dev/null 2>&1; then
    resolved="$(readlink -f "${MODEL_DIR}" 2>/dev/null || true)"
  fi

  if [[ -n "${resolved}" ]]; then
    MODEL_DIR_REAL="$(normalize_path "${resolved}")"
  else
    MODEL_DIR_REAL="${MODEL_DIR}"
  fi

  MODEL_EXTRA_MOUNT=""
  case "${MODEL_DIR_REAL}" in
    "${MODELS_ROOT}"/*) ;;
    *)
      local real_parent
      real_parent="$(dirname "${MODEL_DIR_REAL}")"
      MODEL_EXTRA_MOUNT="${real_parent}:${real_parent}:ro"
      ;;
  esac
}

#######################################
# Snapshot / Validation
#######################################
save_env_snapshot() {
  local file="${LOG_DIR}/env_snapshot.txt"
  {
    echo "timestamp=$(ts)"
    echo "run_ts=${RUN_TS}"
    echo "script_name=${SCRIPT_NAME}"
    echo "script_version=${SCRIPT_VERSION}"
    echo "base_dir=${BASE_DIR}"
    echo "log_dir=${LOG_DIR}"
    echo "run_log=${RUN_LOG}"
    echo "pwd=$(pwd)"
    echo "user=$(whoami 2>/dev/null || true)"
    echo "hostname=$(hostname 2>/dev/null || true)"
    echo "image=${IMAGE}"
    echo "models_root=${MODELS_ROOT}"
    echo "hf_cache=${HF_CACHE}"
    echo "container_name=${CONTAINER_NAME}"
    echo "model_dir=${MODEL_DIR}"
    echo "model_dir_real=${MODEL_DIR_REAL}"
    echo "model_name=${MODEL_NAME}"
    echo "host_port=${HOST_PORT}"
    echo "container_port=${CONTAINER_PORT}"
    echo "gpu_devices=${GPU_DEVICES}"
    echo "tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
    echo "dtype=${DTYPE}"
    echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    echo "max_model_len=${MAX_MODEL_LEN}"
    echo "max_num_seqs=${MAX_NUM_SEQS}"
    echo "tool_call_parser=${TOOL_CALL_PARSER}"
    echo "reasoning_parser=${REASONING_PARSER}"
    echo "enable_auto_tool_choice=${ENABLE_AUTO_TOOL_CHOICE}"
    echo "enable_reasoning=${ENABLE_REASONING}"
    echo "language_model_only=${LANGUAGE_MODEL_ONLY}"
    echo "chat_template=${CHAT_TEMPLATE}"
    echo "trust_remote_code=${TRUST_REMOTE_CODE}"
    echo "disable_custom_all_reduce=${DISABLE_CUSTOM_ALL_REDUCE}"
    echo "enforce_eager=${ENFORCE_EAGER}"
    echo
    echo "### uname -a ###"
    uname -a || true
    echo
    echo "### /etc/os-release ###"
    cat /etc/os-release || true
    echo
    echo "### docker version ###"
    ${DOCKER_BIN} version || true
    echo
    echo "### docker info ###"
    ${DOCKER_BIN} info || true
    echo
    echo "### nvidia-smi ###"
    nvidia-smi || true
    echo
    echo "### df -h ###"
    df -h || true
    echo
    echo "### free -h ###"
    free -h || true
  } >"${file}" 2>&1 || true
  log "Environment snapshot saved: ${file}"
}

check_paths() {
  resolve_model_dir

  [[ -d "${MODELS_ROOT}" ]] || { err "Models root not found: ${MODELS_ROOT}"; return 1; }
  [[ -d "${HF_CACHE}" ]] || { err "HF cache dir not found: ${HF_CACHE}"; return 1; }
  [[ -d "${MODEL_DIR}" ]] || { err "Model dir not found: ${MODEL_DIR}"; return 1; }
  [[ -d "${MODEL_DIR_REAL}" ]] || { err "Resolved model dir not found: ${MODEL_DIR_REAL}"; return 1; }

  if [[ ! -f "${MODEL_DIR_REAL}/config.json" && ! -f "${MODEL_DIR_REAL}/params.json" ]]; then
    err "MODEL_DIR does not look like a valid model root: ${MODEL_DIR_REAL}"
    err "Expected config.json (HF) or params.json (Mistral-style)"
    return 1
  fi

  log "Path check passed"
  log "  MODELS_ROOT = ${MODELS_ROOT}"
  log "  HF_CACHE    = ${HF_CACHE}"
  log "  MODEL_DIR   = ${MODEL_DIR}"
  log "  MODEL_DIR_REAL = ${MODEL_DIR_REAL}"
  if [[ -n "${MODEL_EXTRA_MOUNT}" ]]; then
    log "  MODEL_EXTRA_MOUNT enabled for resolved symlink target"
  fi
}

validate_image() {
  if ${DOCKER_BIN} image inspect "${IMAGE}" >/dev/null 2>&1; then
    log "Docker image exists locally: ${IMAGE}"
  else
    warn "Docker image not found locally: ${IMAGE}"
    warn "You may need: docker pull ${IMAGE}"
  fi
}

validate_ports() {
  divider
  log "Checking port usage"

  if is_port_listening "${HOST_PORT}"; then
    warn "Port ${HOST_PORT} is already listening"
    port_check "${HOST_PORT}" | tee -a "${RUN_LOG}" || true
  else
    log "Port ${HOST_PORT} is free"
  fi
}

validate_gpu() {
  divider
  log "Checking GPU visibility"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>&1 | tee -a "${RUN_LOG}" >/dev/null || true
    nvidia-smi -L 2>&1 | tee -a "${RUN_LOG}" || true
  else
    warn "nvidia-smi not found"
  fi
}

validate_runtime() {
  divider
  log "Checking Docker runtime"
  ${DOCKER_BIN} version 2>&1 | tee -a "${RUN_LOG}" >/dev/null || {
    err "docker version failed"
    return 1
  }
  ${DOCKER_BIN} info 2>&1 | tee -a "${RUN_LOG}" >/dev/null || {
    err "docker info failed"
    return 1
  }
  log "Docker runtime check passed"
}

validate_all() {
  require_bin "${DOCKER_BIN}"
  require_bin "${CURL_BIN}"
  resolve_model_dir

  divider
  log "Validate start"
  save_env_snapshot
  validate_runtime
  validate_gpu
  check_paths
  validate_image
  validate_ports
  divider
  log "Validate finished"
}

#######################################
# Docker Run Command Builder
# 这里用“位置参数”传模型目录，避免 --model 提示
# Docker CLI 对 device=0,1,2,3 这种值要求保留内部引号，否则会把逗号拆成额外字段。
#######################################
build_run_cmd() {
  resolve_model_dir
  local -a cmd
  cmd=(
    "${DOCKER_BIN}" run -d
    --name "${CONTAINER_NAME}"
    --restart unless-stopped
    --gpus "\"${GPU_DEVICES}\""
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --log-opt max-size=100m
    --log-opt max-file=3
    -p "${HOST_PORT}:${CONTAINER_PORT}"
    -v "${MODELS_ROOT}:${MODELS_ROOT}:ro"
    -v "${HF_CACHE}:${HF_CACHE}"
    -e "HF_HOME=${HF_CACHE}"
    "${IMAGE}"
    "${MODEL_DIR_REAL}"
    --host 0.0.0.0
    --port "${CONTAINER_PORT}"
    --served-model-name "${MODEL_NAME}"
    --api-key "${API_KEY}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --dtype "${DTYPE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
  )

  if [[ -n "${MODEL_EXTRA_MOUNT}" ]]; then
    cmd+=(-v "${MODEL_EXTRA_MOUNT}")
  fi

  if [[ -n "${TOOL_CALL_PARSER}" ]]; then
    cmd+=(--tool-call-parser "${TOOL_CALL_PARSER}")
  fi

  if [[ -n "${REASONING_PARSER}" ]]; then
    cmd+=(--reasoning-parser "${REASONING_PARSER}")
  fi

  if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
    cmd+=(--enable-auto-tool-choice)
  fi

  if [[ "${ENABLE_REASONING}" == "1" ]]; then
    cmd+=(--enable-reasoning)
  fi

  if [[ -n "${CHAT_TEMPLATE}" ]]; then
    cmd+=(--chat-template "${CHAT_TEMPLATE}")
  fi

  if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
    cmd+=(--trust-remote-code)
  fi

  if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
    cmd+=(--disable-custom-all-reduce)
  fi

  if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    cmd+=(--enforce-eager)
  fi

  if [[ "${LANGUAGE_MODEL_ONLY}" == "1" ]]; then
    cmd+=(--language-model-only)
  fi

  printf '%q ' "${cmd[@]}"
  printf '\n'
}

#######################################
# Health Check
#######################################
probe_api() {
  ${CURL_BIN} -fsS \
    --connect-timeout "${HEALTH_CONNECT_TIMEOUT}" \
    --max-time "${HEALTH_MAX_TIME}" \
    "http://127.0.0.1:${HOST_PORT}/v1/models" \
    -H "Authorization: Bearer ${API_KEY}"
}

probe_health_endpoint() {
  ${CURL_BIN} -fsS \
    --connect-timeout "${HEALTH_CONNECT_TIMEOUT}" \
    --max-time "${HEALTH_MAX_TIME}" \
    "http://127.0.0.1:${HOST_PORT}/health"
}

probe_ping_endpoint() {
  ${CURL_BIN} -fsS \
    --connect-timeout "${HEALTH_CONNECT_TIMEOUT}" \
    --max-time "${HEALTH_MAX_TIME}" \
    "http://127.0.0.1:${HOST_PORT}/ping"
}

status_probe_result() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    printf '%-8s: READY\n' "${label}"
  else
    printf '%-8s: FAIL\n' "${label}"
  fi
}

container_logs_show_ready_markers() {
  ${DOCKER_BIN} logs --tail 200 "${CONTAINER_NAME}" 2>&1 | grep -Eq \
    'Application startup complete|Starting vLLM API server|Route: /v1/models'
}

health_check() {
  local retries="${1:-${HEALTH_RETRIES}}"
  local interval="${2:-${HEALTH_INTERVAL}}"
  local out_file="${LOG_DIR}/healthcheck.txt"
  local status=""
  local rc=0
  local api_rc=0
  local health_rc=0
  local ping_rc=0
  local log_ready=no

  log "Checking health for ${CONTAINER_NAME} on port ${HOST_PORT} (timeout=$((retries * interval))s)"

  {
    echo "### HEALTHCHECK_TIMESTAMP ###"
    echo "$(ts)"
    echo "### TARGET ###"
    echo "http://127.0.0.1:${HOST_PORT}/v1/models"
    echo "### SETTINGS ###"
    echo "retries=${retries}"
    echo "interval=${interval}"
    echo "connect_timeout=${HEALTH_CONNECT_TIMEOUT}"
    echo "max_time=${HEALTH_MAX_TIME}"
    echo
  } > "${out_file}"

  local i
  for ((i=1; i<=retries; i++)); do
    status="$(${DOCKER_BIN} inspect --format '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo unknown)"
    if [[ "${status}" != "running" ]]; then
      err "${CONTAINER_NAME} is not running during health check (status=${status})"
      collect_failure_context "container_not_running_during_healthcheck"
      return 1
    fi

    {
      echo "attempt=${i} timestamp=$(ts)"
      echo "container_status=${status}"
      api_rc=0
      health_rc=0
      ping_rc=0
      log_ready=no

      if probe_api >/dev/null; then
        api_rc=0
      else
        api_rc=$?
      fi

      if probe_health_endpoint >/dev/null; then
        health_rc=0
      else
        health_rc=$?
      fi

      if probe_ping_endpoint >/dev/null; then
        ping_rc=0
      else
        ping_rc=$?
      fi

      if container_logs_show_ready_markers; then
        log_ready=yes
      fi

      if [[ "${api_rc}" -eq 0 || "${health_rc}" -eq 0 || "${ping_rc}" -eq 0 ]]; then
        rc=0
      else
        rc=1
      fi

      echo "api_exit_code=${api_rc}"
      echo "health_exit_code=${health_rc}"
      echo "ping_exit_code=${ping_rc}"
      echo "log_ready_markers=${log_ready}"
      echo
      echo "exit_code=${rc}"
      echo "----------------------------------------"
    } >> "${out_file}" 2>&1 || true

    if (( i % 10 == 0 )); then
      log "Health check still waiting: attempt=${i}/${retries}, container_status=${status}, api=${api_rc}, health=${health_rc}, ping=${ping_rc}, log_ready=${log_ready}"
    fi

    if [[ "${rc}" -eq 0 ]]; then
      log "${CONTAINER_NAME} is ready"
      return 0
    fi

    sleep "${interval}"
  done

  err "${CONTAINER_NAME} health check failed after ${retries} attempts"
  collect_failure_context "health_check_timeout"
  return 1
}

#######################################
# Failure Context + Summary
#######################################
collect_failure_context() {
  local failure_reason="${1:-unknown}"
  local outdir="${LOG_DIR}/failure_${CONTAINER_NAME}"
  mkdir -p "${outdir}"

  warn "Collecting failure context for ${CONTAINER_NAME} into ${outdir}"

  capture_cmd "${outdir}/docker_ps.txt" ${DOCKER_BIN} ps -a
  capture_cmd "${outdir}/docker_ps_current.txt" ${DOCKER_BIN} ps
  capture_cmd "${outdir}/docker_inspect.txt" ${DOCKER_BIN} inspect "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_inspect_state.txt" ${DOCKER_BIN} inspect --format '{{.State.Status}} restart_count={{.RestartCount}} exit_code={{.State.ExitCode}} oom_killed={{.State.OOMKilled}} error={{.State.Error}}' "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_inspect_devices.txt" ${DOCKER_BIN} inspect --format '{{json .HostConfig.DeviceRequests}}' "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_logs_tail_200.txt" ${DOCKER_BIN} logs --timestamps --tail 200 "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_logs_tail_500.txt" ${DOCKER_BIN} logs --timestamps --tail 500 "${CONTAINER_NAME}"
  capture_cmd "${outdir}/port_check.txt" bash -lc "$(declare -f port_check); port_check ${HOST_PORT}"
  capture_cmd "${outdir}/curl_models.txt" ${CURL_BIN} -sS "http://127.0.0.1:${HOST_PORT}/v1/models" -H "Authorization: Bearer ${API_KEY}"
  capture_cmd "${outdir}/nvidia_smi.txt" nvidia-smi
  capture_cmd "${outdir}/df_h.txt" df -h
  capture_cmd "${outdir}/free_h.txt" free -h
  capture_cmd "${outdir}/models_root_ls.txt" ls -lah "${MODELS_ROOT}"
  capture_cmd "${outdir}/model_dir_ls.txt" ls -lah "${MODEL_DIR}"
  capture_cmd "${outdir}/hf_cache_ls.txt" ls -lah "${HF_CACHE}"
  capture_cmd "${outdir}/docker_info.txt" ${DOCKER_BIN} info
  capture_cmd "${outdir}/docker_version.txt" ${DOCKER_BIN} version

  local cmd_file="${outdir}/start_command.txt"
  build_run_cmd > "${cmd_file}"
  chmod +x "${cmd_file}" || true

  tail -n 200 "${RUN_LOG}" > "${outdir}/script_log_tail_200.txt" 2>/dev/null || true

  local summary_file="${outdir}/error_summary.txt"
  {
    echo "==================== META ===================="
    echo "summary_generated_at=$(ts)"
    echo "run_ts=${RUN_TS}"
    echo "run_log=${RUN_LOG}"
    echo "base_dir=${BASE_DIR}"
    echo "script_path=${BASE_DIR}/${SCRIPT_NAME}"
    echo "script_version=${SCRIPT_VERSION}"
    echo "failure_reason=${failure_reason}"
    echo "container_name=${CONTAINER_NAME}"
    echo "model_dir=${MODEL_DIR}"
    echo "model_dir_real=${MODEL_DIR_REAL}"
    echo "model_name=${MODEL_NAME}"
    echo "image=${IMAGE}"
    echo "models_root=${MODELS_ROOT}"
    echo "hf_cache=${HF_CACHE}"
    echo "host_port=${HOST_PORT}"
    echo "container_port=${CONTAINER_PORT}"
    echo "gpu_devices=${GPU_DEVICES}"
    echo "tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
    echo "dtype=${DTYPE}"
    echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    echo "max_model_len=${MAX_MODEL_LEN}"
    echo "max_num_seqs=${MAX_NUM_SEQS}"
    echo "tool_call_parser=${TOOL_CALL_PARSER}"
    echo "reasoning_parser=${REASONING_PARSER}"
    echo "enable_auto_tool_choice=${ENABLE_AUTO_TOOL_CHOICE}"
    echo "enable_reasoning=${ENABLE_REASONING}"
    echo "language_model_only=${LANGUAGE_MODEL_ONLY}"
    echo "chat_template=${CHAT_TEMPLATE}"
    echo "trust_remote_code=${TRUST_REMOTE_CODE}"
    echo "disable_custom_all_reduce=${DISABLE_CUSTOM_ALL_REDUCE}"
    echo "enforce_eager=${ENFORCE_EAGER}"
    echo "model_extra_mount=$(printf '%s' "${MODEL_EXTRA_MOUNT}")"
    echo

    echo "==================== start_command.txt ===================="
    cat "${cmd_file}" 2>/dev/null || true
    echo

    echo "==================== script_log_tail_200.txt ===================="
    cat "${outdir}/script_log_tail_200.txt" 2>/dev/null || true
    echo

    echo "==================== docker_logs_tail_200.txt ===================="
    cat "${outdir}/docker_logs_tail_200.txt" 2>/dev/null || true
    echo

    echo "==================== docker_inspect_state.txt ===================="
    cat "${outdir}/docker_inspect_state.txt" 2>/dev/null || true
    echo

    echo "==================== docker_inspect_devices.txt ===================="
    cat "${outdir}/docker_inspect_devices.txt" 2>/dev/null || true
    echo

    echo "==================== docker_inspect.txt ===================="
    cat "${outdir}/docker_inspect.txt" 2>/dev/null || true
    echo

    echo "==================== docker_ps.txt ===================="
    cat "${outdir}/docker_ps.txt" 2>/dev/null || true
    echo

    echo "==================== curl_models.txt ===================="
    cat "${outdir}/curl_models.txt" 2>/dev/null || true
    echo

    echo "==================== healthcheck.txt ===================="
    cat "${LOG_DIR}/healthcheck.txt" 2>/dev/null || true
    echo

    echo "==================== port_check.txt ===================="
    cat "${outdir}/port_check.txt" 2>/dev/null || true
    echo

    echo "==================== nvidia_smi.txt ===================="
    cat "${outdir}/nvidia_smi.txt" 2>/dev/null || true
    echo

    echo "==================== df_h.txt ===================="
    cat "${outdir}/df_h.txt" 2>/dev/null || true
    echo

    echo "==================== free_h.txt ===================="
    cat "${outdir}/free_h.txt" 2>/dev/null || true
    echo

    echo "==================== models_root_ls.txt ===================="
    cat "${outdir}/models_root_ls.txt" 2>/dev/null || true
    echo

    echo "==================== model_dir_ls.txt ===================="
    cat "${outdir}/model_dir_ls.txt" 2>/dev/null || true
    echo

    echo "==================== hf_cache_ls.txt ===================="
    cat "${outdir}/hf_cache_ls.txt" 2>/dev/null || true
    echo

    echo "==================== docker_info.txt ===================="
    cat "${outdir}/docker_info.txt" 2>/dev/null || true
    echo

    echo "==================== docker_version.txt ===================="
    cat "${outdir}/docker_version.txt" 2>/dev/null || true
    echo
  } > "${summary_file}"

  warn "Failure context collected: ${outdir}"
  warn "Failure summary generated: ${summary_file}"
  warn "Send this file for analysis: ${summary_file}"
}

#######################################
# Debug Bundle
#######################################
debug_service() {
  local outdir="${LOG_DIR}/debug_${CONTAINER_NAME}"
  mkdir -p "${outdir}"

  log "Collecting debug bundle -> ${outdir}"

  capture_cmd "${outdir}/docker_ps_a.txt" ${DOCKER_BIN} ps -a
  capture_cmd "${outdir}/docker_ps.txt" ${DOCKER_BIN} ps
  capture_cmd "${outdir}/docker_inspect.txt" ${DOCKER_BIN} inspect "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_inspect_state.txt" ${DOCKER_BIN} inspect --format '{{.State.Status}} restart_count={{.RestartCount}} exit_code={{.State.ExitCode}} oom_killed={{.State.OOMKilled}} error={{.State.Error}}' "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_logs_tail_200.txt" ${DOCKER_BIN} logs --timestamps --tail 200 "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_logs_tail_500.txt" ${DOCKER_BIN} logs --timestamps --tail 500 "${CONTAINER_NAME}"
  capture_cmd "${outdir}/port_check.txt" bash -lc "$(declare -f port_check); port_check ${HOST_PORT}"
  capture_cmd "${outdir}/curl_models.txt" ${CURL_BIN} -sS "http://127.0.0.1:${HOST_PORT}/v1/models" -H "Authorization: Bearer ${API_KEY}"
  capture_cmd "${outdir}/nvidia_smi.txt" nvidia-smi
  capture_cmd "${outdir}/df_h.txt" df -h
  capture_cmd "${outdir}/free_h.txt" free -h
  capture_cmd "${outdir}/models_root_ls.txt" ls -lah "${MODELS_ROOT}"
  capture_cmd "${outdir}/model_dir_ls.txt" ls -lah "${MODEL_DIR}"
  capture_cmd "${outdir}/hf_cache_ls.txt" ls -lah "${HF_CACHE}"
  capture_cmd "${outdir}/docker_info.txt" ${DOCKER_BIN} info
  capture_cmd "${outdir}/docker_version.txt" ${DOCKER_BIN} version

  local summary_file="${outdir}/debug_summary.txt"
  {
    echo "==================== META ===================="
    echo "summary_generated_at=$(ts)"
    echo "run_ts=${RUN_TS}"
    echo "run_log=${RUN_LOG}"
    echo "container_name=${CONTAINER_NAME}"
    echo "model_dir=${MODEL_DIR}"
    echo "model_name=${MODEL_NAME}"
    echo

    echo "==================== docker_logs_tail_200.txt ===================="
    cat "${outdir}/docker_logs_tail_200.txt" 2>/dev/null || true
    echo

    echo "==================== docker_inspect_state.txt ===================="
    cat "${outdir}/docker_inspect_state.txt" 2>/dev/null || true
    echo

    echo "==================== docker_inspect.txt ===================="
    cat "${outdir}/docker_inspect.txt" 2>/dev/null || true
    echo

    echo "==================== curl_models.txt ===================="
    cat "${outdir}/curl_models.txt" 2>/dev/null || true
    echo

    echo "==================== nvidia_smi.txt ===================="
    cat "${outdir}/nvidia_smi.txt" 2>/dev/null || true
    echo
  } > "${summary_file}"

  log "Debug bundle created: ${outdir}"
  log "Debug summary created: ${summary_file}"
}

#######################################
# Start / Stop / Restart
#######################################
start_service() {
  divider
  log "Starting container: ${CONTAINER_NAME}"
  remove_container_if_exists

  local cmd_file="${LOG_DIR}/start_command.sh"
  build_run_cmd > "${cmd_file}"
  chmod +x "${cmd_file}" || true

  log "Docker command saved: ${cmd_file}"
  log "Docker command:"
  cat "${cmd_file}" | tee -a "${RUN_LOG}"

  if ! bash "${cmd_file}" 2>&1 | tee -a "${RUN_LOG}"; then
    err "Docker run failed"
    collect_failure_context "docker_run_failed"
    return 1
  fi

  log "Container started in background: ${CONTAINER_NAME}"
  health_check
}

stop_service() {
  divider
  log "Stop requested for container: ${CONTAINER_NAME}"
  stop_container_if_exists
}

restart_service() {
  divider
  log "Restarting container: ${CONTAINER_NAME}"
  stop_service
  start_service
}

#######################################
# Logs / Status
#######################################
logs_service() {
  ${DOCKER_BIN} logs --timestamps -f "${CONTAINER_NAME}"
}

status_service() {
  resolve_model_dir
  divider
  echo "Script Info" | tee -a "${RUN_LOG}"
  divider
  echo "SCRIPT_NAME    : ${SCRIPT_NAME}" | tee -a "${RUN_LOG}"
  echo "SCRIPT_VERSION : ${SCRIPT_VERSION}" | tee -a "${RUN_LOG}"
  echo "BASE_DIR       : ${BASE_DIR}" | tee -a "${RUN_LOG}"
  echo "LOG_DIR        : ${LOG_DIR}" | tee -a "${RUN_LOG}"
  echo "IMAGE          : ${IMAGE}" | tee -a "${RUN_LOG}"
  echo "MODEL_DIR      : ${MODEL_DIR}" | tee -a "${RUN_LOG}"
  echo "MODEL_DIR_REAL : ${MODEL_DIR_REAL}" | tee -a "${RUN_LOG}"
  echo "MODEL_NAME     : ${MODEL_NAME}" | tee -a "${RUN_LOG}"
  echo "MAX_NUM_SEQS   : ${MAX_NUM_SEQS}" | tee -a "${RUN_LOG}"
  echo "TOOL_CALL_PARSER : ${TOOL_CALL_PARSER}" | tee -a "${RUN_LOG}"
  echo "REASONING_PARSER : ${REASONING_PARSER}" | tee -a "${RUN_LOG}"
  echo "ENABLE_AUTO_TOOL_CHOICE : ${ENABLE_AUTO_TOOL_CHOICE}" | tee -a "${RUN_LOG}"
  echo "ENABLE_REASONING : ${ENABLE_REASONING}" | tee -a "${RUN_LOG}"
  echo "CHAT_TEMPLATE : ${CHAT_TEMPLATE}" | tee -a "${RUN_LOG}"
  echo "LANGUAGE_MODEL_ONLY : ${LANGUAGE_MODEL_ONLY}" | tee -a "${RUN_LOG}"
  echo "TRUST_REMOTE_CODE : ${TRUST_REMOTE_CODE}" | tee -a "${RUN_LOG}"
  echo "DISABLE_CUSTOM_ALL_REDUCE : ${DISABLE_CUSTOM_ALL_REDUCE}" | tee -a "${RUN_LOG}"
  echo "ENFORCE_EAGER : ${ENFORCE_EAGER}" | tee -a "${RUN_LOG}"

  echo | tee -a "${RUN_LOG}"
  divider
  echo "Container Status" | tee -a "${RUN_LOG}"
  divider
  ${DOCKER_BIN} ps -a --filter "name=^/${CONTAINER_NAME}$" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}" | tee -a "${RUN_LOG}"

  echo | tee -a "${RUN_LOG}"
  divider
  echo "Probe Status" | tee -a "${RUN_LOG}"
  divider
  if container_running; then
    status_probe_result "models" probe_api | tee -a "${RUN_LOG}"
    status_probe_result "health" probe_health_endpoint | tee -a "${RUN_LOG}"
    status_probe_result "ping" probe_ping_endpoint | tee -a "${RUN_LOG}"
    if probe_api >/dev/null 2>&1 || probe_health_endpoint >/dev/null 2>&1 || probe_ping_endpoint >/dev/null 2>&1; then
      echo "CONTROL  : READY   (http://127.0.0.1:${HOST_PORT})" | tee -a "${RUN_LOG}"
    else
      echo "CONTROL  : NOT READY / STARTING (http://127.0.0.1:${HOST_PORT})" | tee -a "${RUN_LOG}"
    fi
    echo "INFER    : NOT VERIFIED (status only checks endpoints, not generation)" | tee -a "${RUN_LOG}"
  else
    echo "CONTROL  : STOPPED" | tee -a "${RUN_LOG}"
    echo "INFER    : NOT VERIFIED" | tee -a "${RUN_LOG}"
  fi

  echo | tee -a "${RUN_LOG}"
  divider
  echo "Port Listening" | tee -a "${RUN_LOG}"
  divider
  port_check "${HOST_PORT}" | tee -a "${RUN_LOG}" || true

  echo | tee -a "${RUN_LOG}"
  divider
  echo "GPU Status" | tee -a "${RUN_LOG}"
  divider
  nvidia-smi 2>&1 | tee -a "${RUN_LOG}" || true
}

#######################################
# Main
#######################################
main() {
  require_bin "${DOCKER_BIN}"
  require_bin "${CURL_BIN}"

  local cmd="${1:-}"

  log "Script start: ${SCRIPT_NAME} ${*:-}"
  log "Run log file: ${RUN_LOG}"
  log "Run directory: ${LOG_DIR}"

  case "${cmd}" in
    validate)
      validate_all
      ;;
    startup)
      validate_all
      start_service
      ;;
    stop)
      stop_service
      ;;
    restart)
      validate_all
      restart_service
      ;;
    status)
      status_service
      ;;
    logs)
      logs_service
      ;;
    debug)
      debug_service
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
