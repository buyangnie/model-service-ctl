#!/usr/bin/env bash
set -euo pipefail

#######################################
# 通用 vLLM 容器管理脚本（双服务版）
# 当前默认环境：
#   - 脚本目录: /opsfactory/model_service
#   - 生成模型目录: /data/models/Qwen3-14B
#   - Embedding 模型目录: /data/models/Qwen3-Embedding-4B
#######################################

#######################################
# Script Meta
#######################################
SCRIPT_NAME="$(basename "$0")"
SCRIPT_VERSION="1.0"
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
IMAGE="${IMAGE:-vllm/vllm-openai:v0.17.0-x86_64}"
API_KEY="${API_KEY:-change-me}"

MODELS_ROOT="${MODELS_ROOT:-/data/models}"
HF_CACHE="${HF_CACHE:-/data/hf_cache}"

HEALTH_RETRIES="${HEALTH_RETRIES:-600}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-2}"
HEALTH_CONNECT_TIMEOUT="${HEALTH_CONNECT_TIMEOUT:-2}"
HEALTH_MAX_TIME="${HEALTH_MAX_TIME:-8}"

CURL_BIN="${CURL_BIN:-curl}"
DOCKER_BIN="${DOCKER_BIN:-docker}"

QWEN3_14B_CONTAINER_NAME="${QWEN3_14B_CONTAINER_NAME:-vllm-qwen3-14b}"
QWEN3_14B_MODEL_DIR="${QWEN3_14B_MODEL_DIR:-/data/models/Qwen3-14B}"
QWEN3_14B_MODEL_NAME="${QWEN3_14B_MODEL_NAME:-Qwen3-14B}"
QWEN3_14B_HOST_PORT="${QWEN3_14B_HOST_PORT:-8000}"
QWEN3_14B_CONTAINER_PORT="${QWEN3_14B_CONTAINER_PORT:-8000}"
QWEN3_14B_GPU_DEVICES="${QWEN3_14B_GPU_DEVICES:-device=0,1}"
QWEN3_14B_TENSOR_PARALLEL_SIZE="${QWEN3_14B_TENSOR_PARALLEL_SIZE:-2}"
QWEN3_14B_DTYPE="${QWEN3_14B_DTYPE:-float16}"
QWEN3_14B_GPU_MEMORY_UTILIZATION="${QWEN3_14B_GPU_MEMORY_UTILIZATION:-0.85}"
QWEN3_14B_MAX_MODEL_LEN="${QWEN3_14B_MAX_MODEL_LEN:-40960}"
QWEN3_14B_MAX_NUM_SEQS="${QWEN3_14B_MAX_NUM_SEQS:-1}"
QWEN3_14B_TOOL_CALL_PARSER="${QWEN3_14B_TOOL_CALL_PARSER:-hermes}"
QWEN3_14B_ENABLE_AUTO_TOOL_CHOICE="${QWEN3_14B_ENABLE_AUTO_TOOL_CHOICE:-1}"
QWEN3_14B_LANGUAGE_MODEL_ONLY="${QWEN3_14B_LANGUAGE_MODEL_ONLY:-0}"
QWEN3_14B_THINKING_MODE="${QWEN3_14B_THINKING_MODE:-think}"
QWEN3_14B_CHAT_TEMPLATE="${QWEN3_14B_CHAT_TEMPLATE:-}"
QWEN3_14B_TRUST_REMOTE_CODE="${QWEN3_14B_TRUST_REMOTE_CODE:-0}"
QWEN3_14B_DISABLE_CUSTOM_ALL_REDUCE="${QWEN3_14B_DISABLE_CUSTOM_ALL_REDUCE:-1}"
QWEN3_14B_ENFORCE_EAGER="${QWEN3_14B_ENFORCE_EAGER:-1}"
QWEN3_14B_RUNNER="${QWEN3_14B_RUNNER:-}"

QWEN3_EMBEDDING_4B_CONTAINER_NAME="${QWEN3_EMBEDDING_4B_CONTAINER_NAME:-vllm-qwen3-embedding-4b}"
QWEN3_EMBEDDING_4B_MODEL_DIR="${QWEN3_EMBEDDING_4B_MODEL_DIR:-/data/models/Qwen3-Embedding-4B}"
QWEN3_EMBEDDING_4B_MODEL_NAME="${QWEN3_EMBEDDING_4B_MODEL_NAME:-Qwen3-Embedding-4B}"
QWEN3_EMBEDDING_4B_HOST_PORT="${QWEN3_EMBEDDING_4B_HOST_PORT:-8001}"
QWEN3_EMBEDDING_4B_CONTAINER_PORT="${QWEN3_EMBEDDING_4B_CONTAINER_PORT:-8000}"
QWEN3_EMBEDDING_4B_GPU_DEVICES="${QWEN3_EMBEDDING_4B_GPU_DEVICES:-device=3}"
QWEN3_EMBEDDING_4B_TENSOR_PARALLEL_SIZE="${QWEN3_EMBEDDING_4B_TENSOR_PARALLEL_SIZE:-1}"
QWEN3_EMBEDDING_4B_DTYPE="${QWEN3_EMBEDDING_4B_DTYPE:-float16}"
QWEN3_EMBEDDING_4B_GPU_MEMORY_UTILIZATION="${QWEN3_EMBEDDING_4B_GPU_MEMORY_UTILIZATION:-0.85}"
QWEN3_EMBEDDING_4B_MAX_MODEL_LEN="${QWEN3_EMBEDDING_4B_MAX_MODEL_LEN:-32768}"
QWEN3_EMBEDDING_4B_MAX_NUM_SEQS="${QWEN3_EMBEDDING_4B_MAX_NUM_SEQS:-8}"
QWEN3_EMBEDDING_4B_TOOL_CALL_PARSER="${QWEN3_EMBEDDING_4B_TOOL_CALL_PARSER:-}"
QWEN3_EMBEDDING_4B_ENABLE_AUTO_TOOL_CHOICE="${QWEN3_EMBEDDING_4B_ENABLE_AUTO_TOOL_CHOICE:-0}"
QWEN3_EMBEDDING_4B_LANGUAGE_MODEL_ONLY="${QWEN3_EMBEDDING_4B_LANGUAGE_MODEL_ONLY:-0}"
QWEN3_EMBEDDING_4B_THINKING_MODE="${QWEN3_EMBEDDING_4B_THINKING_MODE:-no_think}"
QWEN3_EMBEDDING_4B_CHAT_TEMPLATE="${QWEN3_EMBEDDING_4B_CHAT_TEMPLATE:-}"
QWEN3_EMBEDDING_4B_TRUST_REMOTE_CODE="${QWEN3_EMBEDDING_4B_TRUST_REMOTE_CODE:-0}"
QWEN3_EMBEDDING_4B_DISABLE_CUSTOM_ALL_REDUCE="${QWEN3_EMBEDDING_4B_DISABLE_CUSTOM_ALL_REDUCE:-1}"
QWEN3_EMBEDDING_4B_ENFORCE_EAGER="${QWEN3_EMBEDDING_4B_ENFORCE_EAGER:-1}"
QWEN3_EMBEDDING_4B_RUNNER="${QWEN3_EMBEDDING_4B_RUNNER:-pooling}"

CURRENT_SERVICE_ID=""
CURRENT_SERVICE_LABEL=""
CONTAINER_NAME=""
MODEL_DIR=""
MODEL_NAME=""
HOST_PORT=""
CONTAINER_PORT=""
GPU_DEVICES=""
TENSOR_PARALLEL_SIZE=""
DTYPE=""
GPU_MEMORY_UTILIZATION=""
MAX_MODEL_LEN=""
MAX_NUM_SEQS=""
TOOL_CALL_PARSER=""
ENABLE_AUTO_TOOL_CHOICE=""
LANGUAGE_MODEL_ONLY=""
THINKING_MODE=""
CHAT_TEMPLATE=""
TRUST_REMOTE_CODE=""
DISABLE_CUSTOM_ALL_REDUCE=""
ENFORCE_EAGER=""
RUNNER=""
MODEL_DIR_REAL=""
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
  ${SCRIPT_NAME} validate [all|14b|embedding]
  ${SCRIPT_NAME} startup [all|14b|embedding]
  ${SCRIPT_NAME} stop [all|14b|embedding]
  ${SCRIPT_NAME} restart [all|14b|embedding]
  ${SCRIPT_NAME} status [all|14b|embedding]
  ${SCRIPT_NAME} logs [all|14b|embedding]
  ${SCRIPT_NAME} debug [all|14b|embedding]

Targets:
  all       管理两个服务（默认）
  14b       仅管理 Qwen3-14B
  embedding 仅管理 Qwen3-Embedding-4B

Current defaults:
  IMAGE=${IMAGE}
  MODELS_ROOT=${MODELS_ROOT}
  HF_CACHE=${HF_CACHE}

  [14B]
  CONTAINER_NAME=${QWEN3_14B_CONTAINER_NAME}
  MODEL_DIR=${QWEN3_14B_MODEL_DIR}
  MODEL_NAME=${QWEN3_14B_MODEL_NAME}
  HOST_PORT=${QWEN3_14B_HOST_PORT}
  GPU_DEVICES=${QWEN3_14B_GPU_DEVICES}
  TENSOR_PARALLEL_SIZE=${QWEN3_14B_TENSOR_PARALLEL_SIZE}
  MAX_MODEL_LEN=${QWEN3_14B_MAX_MODEL_LEN}
  MAX_NUM_SEQS=${QWEN3_14B_MAX_NUM_SEQS}
  TOOL_CALL_PARSER=${QWEN3_14B_TOOL_CALL_PARSER}
  ENABLE_AUTO_TOOL_CHOICE=${QWEN3_14B_ENABLE_AUTO_TOOL_CHOICE}
  LANGUAGE_MODEL_ONLY=${QWEN3_14B_LANGUAGE_MODEL_ONLY}
  THINKING_MODE=${QWEN3_14B_THINKING_MODE}
  RUNNER=${QWEN3_14B_RUNNER}

  [Embedding]
  CONTAINER_NAME=${QWEN3_EMBEDDING_4B_CONTAINER_NAME}
  MODEL_DIR=${QWEN3_EMBEDDING_4B_MODEL_DIR}
  MODEL_NAME=${QWEN3_EMBEDDING_4B_MODEL_NAME}
  HOST_PORT=${QWEN3_EMBEDDING_4B_HOST_PORT}
  GPU_DEVICES=${QWEN3_EMBEDDING_4B_GPU_DEVICES}
  TENSOR_PARALLEL_SIZE=${QWEN3_EMBEDDING_4B_TENSOR_PARALLEL_SIZE}
  MAX_MODEL_LEN=${QWEN3_EMBEDDING_4B_MAX_MODEL_LEN}
  MAX_NUM_SEQS=${QWEN3_EMBEDDING_4B_MAX_NUM_SEQS}
  TOOL_CALL_PARSER=${QWEN3_EMBEDDING_4B_TOOL_CALL_PARSER}
  ENABLE_AUTO_TOOL_CHOICE=${QWEN3_EMBEDDING_4B_ENABLE_AUTO_TOOL_CHOICE}
  LANGUAGE_MODEL_ONLY=${QWEN3_EMBEDDING_4B_LANGUAGE_MODEL_ONLY}
  THINKING_MODE=${QWEN3_EMBEDDING_4B_THINKING_MODE}
  RUNNER=${QWEN3_EMBEDDING_4B_RUNNER}

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

set_service_context() {
  local service="${1:-}"
  case "${service}" in
    14b)
      CURRENT_SERVICE_ID="14b"
      CURRENT_SERVICE_LABEL="Qwen3-14B"
      CONTAINER_NAME="${QWEN3_14B_CONTAINER_NAME}"
      MODEL_DIR="${QWEN3_14B_MODEL_DIR}"
      MODEL_NAME="${QWEN3_14B_MODEL_NAME}"
      HOST_PORT="${QWEN3_14B_HOST_PORT}"
      CONTAINER_PORT="${QWEN3_14B_CONTAINER_PORT}"
      GPU_DEVICES="${QWEN3_14B_GPU_DEVICES}"
      TENSOR_PARALLEL_SIZE="${QWEN3_14B_TENSOR_PARALLEL_SIZE}"
      DTYPE="${QWEN3_14B_DTYPE}"
      GPU_MEMORY_UTILIZATION="${QWEN3_14B_GPU_MEMORY_UTILIZATION}"
      MAX_MODEL_LEN="${QWEN3_14B_MAX_MODEL_LEN}"
      MAX_NUM_SEQS="${QWEN3_14B_MAX_NUM_SEQS}"
      TOOL_CALL_PARSER="${QWEN3_14B_TOOL_CALL_PARSER}"
      ENABLE_AUTO_TOOL_CHOICE="${QWEN3_14B_ENABLE_AUTO_TOOL_CHOICE}"
      LANGUAGE_MODEL_ONLY="${QWEN3_14B_LANGUAGE_MODEL_ONLY}"
      THINKING_MODE="${QWEN3_14B_THINKING_MODE}"
      CHAT_TEMPLATE="${QWEN3_14B_CHAT_TEMPLATE}"
      TRUST_REMOTE_CODE="${QWEN3_14B_TRUST_REMOTE_CODE}"
      DISABLE_CUSTOM_ALL_REDUCE="${QWEN3_14B_DISABLE_CUSTOM_ALL_REDUCE}"
      ENFORCE_EAGER="${QWEN3_14B_ENFORCE_EAGER}"
      RUNNER="${QWEN3_14B_RUNNER}"
      ;;
    embedding)
      CURRENT_SERVICE_ID="embedding"
      CURRENT_SERVICE_LABEL="Qwen3-Embedding-4B"
      CONTAINER_NAME="${QWEN3_EMBEDDING_4B_CONTAINER_NAME}"
      MODEL_DIR="${QWEN3_EMBEDDING_4B_MODEL_DIR}"
      MODEL_NAME="${QWEN3_EMBEDDING_4B_MODEL_NAME}"
      HOST_PORT="${QWEN3_EMBEDDING_4B_HOST_PORT}"
      CONTAINER_PORT="${QWEN3_EMBEDDING_4B_CONTAINER_PORT}"
      GPU_DEVICES="${QWEN3_EMBEDDING_4B_GPU_DEVICES}"
      TENSOR_PARALLEL_SIZE="${QWEN3_EMBEDDING_4B_TENSOR_PARALLEL_SIZE}"
      DTYPE="${QWEN3_EMBEDDING_4B_DTYPE}"
      GPU_MEMORY_UTILIZATION="${QWEN3_EMBEDDING_4B_GPU_MEMORY_UTILIZATION}"
      MAX_MODEL_LEN="${QWEN3_EMBEDDING_4B_MAX_MODEL_LEN}"
      MAX_NUM_SEQS="${QWEN3_EMBEDDING_4B_MAX_NUM_SEQS}"
      TOOL_CALL_PARSER="${QWEN3_EMBEDDING_4B_TOOL_CALL_PARSER}"
      ENABLE_AUTO_TOOL_CHOICE="${QWEN3_EMBEDDING_4B_ENABLE_AUTO_TOOL_CHOICE}"
      LANGUAGE_MODEL_ONLY="${QWEN3_EMBEDDING_4B_LANGUAGE_MODEL_ONLY}"
      THINKING_MODE="${QWEN3_EMBEDDING_4B_THINKING_MODE}"
      CHAT_TEMPLATE="${QWEN3_EMBEDDING_4B_CHAT_TEMPLATE}"
      TRUST_REMOTE_CODE="${QWEN3_EMBEDDING_4B_TRUST_REMOTE_CODE}"
      DISABLE_CUSTOM_ALL_REDUCE="${QWEN3_EMBEDDING_4B_DISABLE_CUSTOM_ALL_REDUCE}"
      ENFORCE_EAGER="${QWEN3_EMBEDDING_4B_ENFORCE_EAGER}"
      RUNNER="${QWEN3_EMBEDDING_4B_RUNNER}"
      ;;
    *)
      err "Unknown service target: ${service}"
      exit 1
      ;;
  esac

  MODEL_DIR_REAL="${MODEL_DIR}"
  MODEL_EXTRA_MOUNT=""
}

for_each_service() {
  local target="${1:-all}"
  case "${target}" in
    all)
      printf '%s\n' "14b" "embedding"
      ;;
    14b|embedding)
      printf '%s\n' "${target}"
      ;;
    *)
      err "Unknown target: ${target}. Expected: all | 14b | embedding"
      exit 1
      ;;
  esac
}

container_exists() {
  ${DOCKER_BIN} ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

remove_container_if_exists() {
  if container_exists; then
    log "[${CURRENT_SERVICE_ID}] Removing existing container: ${CONTAINER_NAME}"
    ${DOCKER_BIN} rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  else
    log "[${CURRENT_SERVICE_ID}] Container not found, skip remove: ${CONTAINER_NAME}"
  fi
}

stop_container_if_exists() {
  if container_exists; then
    log "[${CURRENT_SERVICE_ID}] Stopping container: ${CONTAINER_NAME}"
    ${DOCKER_BIN} rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  else
    log "[${CURRENT_SERVICE_ID}] Container not found, skip stop: ${CONTAINER_NAME}"
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
# Diagnostics
#######################################
dump_env_snapshot() {
  local outfile="${LOG_DIR}/env_snapshot.txt"
  {
    echo "pwd=$(pwd)"
    echo "user=$(whoami 2>/dev/null || true)"
    echo "hostname=$(hostname 2>/dev/null || true)"
    echo "image=${IMAGE}"
    echo "models_root=${MODELS_ROOT}"
    echo "hf_cache=${HF_CACHE}"
    echo

    for service in 14b embedding; do
      set_service_context "${service}"
      resolve_model_dir
      echo "[${service}]"
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
      echo "enable_auto_tool_choice=${ENABLE_AUTO_TOOL_CHOICE}"
      echo "language_model_only=${LANGUAGE_MODEL_ONLY}"
      echo "thinking_mode=${THINKING_MODE}"
      echo "chat_template=${CHAT_TEMPLATE}"
      echo "trust_remote_code=${TRUST_REMOTE_CODE}"
      echo "disable_custom_all_reduce=${DISABLE_CUSTOM_ALL_REDUCE}"
      echo "enforce_eager=${ENFORCE_EAGER}"
      echo "runner=${RUNNER}"
      echo
    done

    echo "### uname -a ###"
    uname -a || true
    echo
    echo "### /etc/os-release ###"
    cat /etc/os-release || true
  } >"${outfile}" 2>&1 || true
}

#######################################
# Validation
#######################################
validate_current_service() {
  resolve_model_dir

  [[ -d "${MODELS_ROOT}" ]] || { err "[${CURRENT_SERVICE_ID}] Models root not found: ${MODELS_ROOT}"; return 1; }
  [[ -d "${MODEL_DIR}" ]] || { err "[${CURRENT_SERVICE_ID}] Model dir not found: ${MODEL_DIR}"; return 1; }
  [[ -d "${MODEL_DIR_REAL}" ]] || { err "[${CURRENT_SERVICE_ID}] Resolved model dir not found: ${MODEL_DIR_REAL}"; return 1; }

  case "${THINKING_MODE}" in
    think|no_think) ;;
    *)
      err "[${CURRENT_SERVICE_ID}] Invalid THINKING_MODE: ${THINKING_MODE}. Expected: think | no_think"
      return 1
      ;;
  esac

  if [[ ! -f "${MODEL_DIR_REAL}/config.json" && ! -f "${MODEL_DIR_REAL}/params.json" ]]; then
    err "[${CURRENT_SERVICE_ID}] MODEL_DIR does not look like a valid model root: ${MODEL_DIR_REAL}"
    return 1
  fi

  mkdir -p "${HF_CACHE}"

  log "[${CURRENT_SERVICE_ID}] Validation passed:"
  log "[${CURRENT_SERVICE_ID}]   MODEL_DIR = ${MODEL_DIR}"
  log "[${CURRENT_SERVICE_ID}]   MODEL_DIR_REAL = ${MODEL_DIR_REAL}"
  log "[${CURRENT_SERVICE_ID}]   HOST_PORT = ${HOST_PORT}"
  log "[${CURRENT_SERVICE_ID}]   GPU_DEVICES = ${GPU_DEVICES}"
}

validate_env() {
  divider
  log "Validating runtime environment"
  require_bin "${DOCKER_BIN}"
  require_bin "${CURL_BIN}"

  local service
  for service in $(for_each_service "${1:-all}"); do
    set_service_context "${service}"
    validate_current_service
  done
}

#######################################
# Docker Run Command Builder
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

  if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
    cmd+=(--enable-auto-tool-choice)
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

  if [[ -n "${RUNNER}" ]]; then
    cmd+=(--runner "${RUNNER}")
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

container_seems_healthy() {
  if probe_health_endpoint >/dev/null 2>&1; then
    return 0
  fi
  if probe_api >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

collect_failure_context() {
  local failure_reason="${1:-unknown}"
  local outdir="${LOG_DIR}/failure_${CONTAINER_NAME}"
  local cmd_file="${LOG_DIR}/start_command_${CURRENT_SERVICE_ID}.txt"
  mkdir -p "${outdir}"

  warn "[${CURRENT_SERVICE_ID}] Collecting failure context for ${CONTAINER_NAME} into ${outdir}"
  capture_cmd "${outdir}/docker_ps_a.txt" ${DOCKER_BIN} ps -a
  capture_cmd "${outdir}/docker_ps.txt" ${DOCKER_BIN} ps
  capture_cmd "${outdir}/docker_inspect.txt" ${DOCKER_BIN} inspect "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_inspect_state.txt" ${DOCKER_BIN} inspect --format '{{.State.Status}} restart_count={{.RestartCount}} exit_code={{.State.ExitCode}} oom_killed={{.State.OOMKilled}} error={{.State.Error}}' "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_inspect_devices.txt" ${DOCKER_BIN} inspect --format '{{json .HostConfig.DeviceRequests}}' "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_logs_tail_200.txt" ${DOCKER_BIN} logs --timestamps --tail 200 "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_logs_tail_500.txt" ${DOCKER_BIN} logs --timestamps --tail 500 "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_images.txt" ${DOCKER_BIN} images
  capture_cmd "${outdir}/port_check.txt" port_check "${HOST_PORT}"
  capture_cmd "${outdir}/ss_port.txt" bash -lc "command -v ss >/dev/null 2>&1 && ss -lntp | grep -E '[:.]${HOST_PORT}( |$)' || true"
  capture_cmd "${outdir}/nvidia_smi.txt" bash -lc "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true"
  capture_cmd "${outdir}/nvidia_smi_pmon.txt" bash -lc "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi pmon -c 1 || true"
  capture_cmd "${outdir}/model_dir_ls.txt" ls -lah "${MODEL_DIR}"
  capture_cmd "${outdir}/model_dir_real_ls.txt" ls -lah "${MODEL_DIR_REAL}"
  capture_cmd "${outdir}/hf_cache_ls.txt" ls -lah "${HF_CACHE}"

  {
    echo "failure_timestamp=$(ts)"
    echo "script_path=${BASE_DIR}/${SCRIPT_NAME}"
    echo "script_version=${SCRIPT_VERSION}"
    echo "failure_reason=${failure_reason}"
    echo "service_id=${CURRENT_SERVICE_ID}"
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
    echo "enable_auto_tool_choice=${ENABLE_AUTO_TOOL_CHOICE}"
    echo "language_model_only=${LANGUAGE_MODEL_ONLY}"
    echo "thinking_mode=${THINKING_MODE}"
    echo "chat_template=${CHAT_TEMPLATE}"
    echo "trust_remote_code=${TRUST_REMOTE_CODE}"
    echo "disable_custom_all_reduce=${DISABLE_CUSTOM_ALL_REDUCE}"
    echo "enforce_eager=${ENFORCE_EAGER}"
    echo "runner=${RUNNER}"
    echo "model_extra_mount=$(printf '%s' "${MODEL_EXTRA_MOUNT}")"
    echo
    echo "==================== start_command ===================="
    cat "${cmd_file}" 2>/dev/null || true
  } >"${outdir}/summary.txt"

  warn "[${CURRENT_SERVICE_ID}] Failure context ready: ${outdir}"
}

health_check() {
  local retries="${HEALTH_RETRIES}"
  local interval="${HEALTH_INTERVAL}"
  local attempt=1

  log "[${CURRENT_SERVICE_ID}] Checking health for ${CONTAINER_NAME} on port ${HOST_PORT} (timeout=$((retries * interval))s)"
  while (( attempt <= retries )); do
    if container_seems_healthy; then
      log "[${CURRENT_SERVICE_ID}] ${CONTAINER_NAME} is ready"
      return 0
    fi

    local status
    status="$(${DOCKER_BIN} inspect --format '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo unknown)"
    if [[ "${status}" != "running" ]]; then
      err "[${CURRENT_SERVICE_ID}] ${CONTAINER_NAME} is not running during health check (status=${status})"
      collect_failure_context "container_not_running"
      return 1
    fi

    if (( attempt % 10 == 0 )); then
      log "[${CURRENT_SERVICE_ID}] Health check attempt ${attempt}/${retries} not ready yet"
    fi
    sleep "${interval}"
    attempt=$((attempt + 1))
  done

  err "[${CURRENT_SERVICE_ID}] ${CONTAINER_NAME} health check failed after ${retries} attempts"
  collect_failure_context "health_timeout"
  return 1
}

debug_service() {
  resolve_model_dir
  local outdir="${LOG_DIR}/debug_${CONTAINER_NAME}"
  mkdir -p "${outdir}"
  log "[${CURRENT_SERVICE_ID}] Collecting debug context into ${outdir}"
  capture_cmd "${outdir}/docker_ps.txt" ${DOCKER_BIN} ps -a
  capture_cmd "${outdir}/docker_inspect.txt" ${DOCKER_BIN} inspect "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_inspect_state.txt" ${DOCKER_BIN} inspect --format '{{.State.Status}} restart_count={{.RestartCount}} exit_code={{.State.ExitCode}} oom_killed={{.State.OOMKilled}} error={{.State.Error}}' "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_logs_tail_200.txt" ${DOCKER_BIN} logs --timestamps --tail 200 "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_logs_tail_500.txt" ${DOCKER_BIN} logs --timestamps --tail 500 "${CONTAINER_NAME}"
  capture_cmd "${outdir}/nvidia_smi.txt" bash -lc "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true"
  capture_cmd "${outdir}/port_check.txt" port_check "${HOST_PORT}"
  capture_cmd "${outdir}/model_dir_ls.txt" ls -lah "${MODEL_DIR}"

  {
    echo "service_id=${CURRENT_SERVICE_ID}"
    echo "container_name=${CONTAINER_NAME}"
    echo "model_dir=${MODEL_DIR}"
    echo "model_name=${MODEL_NAME}"
    echo "host_port=${HOST_PORT}"
    echo "gpu_devices=${GPU_DEVICES}"
  } >"${outdir}/summary.txt"

  log "[${CURRENT_SERVICE_ID}] Debug bundle written to ${outdir}"
}

#######################################
# Lifecycle
#######################################
start_single_service() {
  resolve_model_dir

  if is_port_listening "${HOST_PORT}"; then
    warn "[${CURRENT_SERVICE_ID}] Host port ${HOST_PORT} already appears to be listening"
    port_check "${HOST_PORT}" | tee -a "${RUN_LOG}" || true
  fi

  remove_container_if_exists

  local cmd_str
  cmd_str="$(build_run_cmd)"
  echo "${cmd_str}" >"${LOG_DIR}/start_command_${CURRENT_SERVICE_ID}.txt"

  log "[${CURRENT_SERVICE_ID}] Starting container: ${CONTAINER_NAME}"
  if ! bash -lc "${cmd_str}" >>"${RUN_LOG}" 2>&1; then
    err "[${CURRENT_SERVICE_ID}] docker run failed"
    collect_failure_context "docker_run_failed"
    return 1
  fi

  log "[${CURRENT_SERVICE_ID}] Container started in background: ${CONTAINER_NAME}"
  health_check
}

start_service() {
  divider
  log "Script start: ${SCRIPT_NAME} startup ${1:-all}"
  log "Run log file: ${RUN_LOG}"
  log "Run directory: ${LOG_DIR}"

  dump_env_snapshot
  validate_env "${1:-all}"

  local service
  for service in $(for_each_service "${1:-all}"); do
    set_service_context "${service}"
    start_single_service
  done
}

stop_service() {
  divider
  local service
  for service in $(for_each_service "${1:-all}"); do
    set_service_context "${service}"
    log "[${CURRENT_SERVICE_ID}] Stop requested for container: ${CONTAINER_NAME}"
    stop_container_if_exists
  done
}

restart_service() {
  divider
  log "Restarting services: ${1:-all}"
  stop_service "${1:-all}"
  start_service "${1:-all}"
}

#######################################
# Logs / Status
#######################################
logs_single_service() {
  ${DOCKER_BIN} logs --timestamps -f "${CONTAINER_NAME}"
}

logs_all_services() {
  local pid1 pid2
  ${DOCKER_BIN} logs --timestamps -f "${QWEN3_14B_CONTAINER_NAME}" 2>&1 | sed "s/^/[14b] /" &
  pid1=$!
  ${DOCKER_BIN} logs --timestamps -f "${QWEN3_EMBEDDING_4B_CONTAINER_NAME}" 2>&1 | sed "s/^/[embedding] /" &
  pid2=$!

  trap 'kill "${pid1}" "${pid2}" 2>/dev/null || true' INT TERM EXIT
  wait "${pid1}" "${pid2}"
}

status_service() {
  divider
  echo "Script Info" | tee -a "${RUN_LOG}"
  divider
  echo "SCRIPT_NAME    : ${SCRIPT_NAME}" | tee -a "${RUN_LOG}"
  echo "SCRIPT_VERSION : ${SCRIPT_VERSION}" | tee -a "${RUN_LOG}"
  echo "BASE_DIR       : ${BASE_DIR}" | tee -a "${RUN_LOG}"
  echo "LOG_DIR        : ${LOG_DIR}" | tee -a "${RUN_LOG}"
  echo "IMAGE          : ${IMAGE}" | tee -a "${RUN_LOG}"

  local service
  for service in $(for_each_service "${1:-all}"); do
    set_service_context "${service}"
    resolve_model_dir
    echo | tee -a "${RUN_LOG}"
    divider
    echo "Service Status: ${CURRENT_SERVICE_LABEL}" | tee -a "${RUN_LOG}"
    divider
    echo "CONTAINER_NAME : ${CONTAINER_NAME}" | tee -a "${RUN_LOG}"
    echo "MODEL_DIR      : ${MODEL_DIR}" | tee -a "${RUN_LOG}"
    echo "MODEL_DIR_REAL : ${MODEL_DIR_REAL}" | tee -a "${RUN_LOG}"
    echo "MODEL_NAME     : ${MODEL_NAME}" | tee -a "${RUN_LOG}"
    echo "HOST_PORT      : ${HOST_PORT}" | tee -a "${RUN_LOG}"
    echo "GPU_DEVICES    : ${GPU_DEVICES}" | tee -a "${RUN_LOG}"
    echo "TENSOR_PARALLEL_SIZE : ${TENSOR_PARALLEL_SIZE}" | tee -a "${RUN_LOG}"
    echo "MAX_NUM_SEQS   : ${MAX_NUM_SEQS}" | tee -a "${RUN_LOG}"
    echo "TOOL_CALL_PARSER : ${TOOL_CALL_PARSER}" | tee -a "${RUN_LOG}"
    echo "ENABLE_AUTO_TOOL_CHOICE : ${ENABLE_AUTO_TOOL_CHOICE}" | tee -a "${RUN_LOG}"
    echo "CHAT_TEMPLATE : ${CHAT_TEMPLATE}" | tee -a "${RUN_LOG}"
    echo "LANGUAGE_MODEL_ONLY : ${LANGUAGE_MODEL_ONLY}" | tee -a "${RUN_LOG}"
    echo "THINKING_MODE : ${THINKING_MODE}" | tee -a "${RUN_LOG}"
    echo "TRUST_REMOTE_CODE : ${TRUST_REMOTE_CODE}" | tee -a "${RUN_LOG}"
    echo "DISABLE_CUSTOM_ALL_REDUCE : ${DISABLE_CUSTOM_ALL_REDUCE}" | tee -a "${RUN_LOG}"
    echo "ENFORCE_EAGER : ${ENFORCE_EAGER}" | tee -a "${RUN_LOG}"
    echo "RUNNER : ${RUNNER}" | tee -a "${RUN_LOG}"

    echo | tee -a "${RUN_LOG}"
    ${DOCKER_BIN} ps -a --filter "name=^/${CONTAINER_NAME}$" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}" | tee -a "${RUN_LOG}"

    echo | tee -a "${RUN_LOG}"
    if container_seems_healthy; then
      echo "Health: READY" | tee -a "${RUN_LOG}"
    else
      echo "Health: NOT_READY" | tee -a "${RUN_LOG}"
    fi
  done
}

#######################################
# Main
#######################################
main() {
  local action="${1:-}"
  local target="${2:-all}"

  case "${action}" in
    validate)
      log "Script start: ${SCRIPT_NAME} validate ${target}"
      log "Run log file: ${RUN_LOG}"
      log "Run directory: ${LOG_DIR}"
      dump_env_snapshot
      validate_env "${target}"
      ;;
    startup)
      start_service "${target}"
      ;;
    stop)
      log "Script start: ${SCRIPT_NAME} stop ${target}"
      log "Run log file: ${RUN_LOG}"
      log "Run directory: ${LOG_DIR}"
      stop_service "${target}"
      ;;
    restart)
      restart_service "${target}"
      ;;
    status)
      log "Script start: ${SCRIPT_NAME} status ${target}"
      log "Run log file: ${RUN_LOG}"
      log "Run directory: ${LOG_DIR}"
      status_service "${target}"
      ;;
    logs)
      log "Script start: ${SCRIPT_NAME} logs ${target}"
      log "Run log file: ${RUN_LOG}"
      log "Run directory: ${LOG_DIR}"
      if [[ "${target}" == "all" ]]; then
        logs_all_services
      else
        set_service_context "${target}"
        logs_single_service
      fi
      ;;
    debug)
      log "Script start: ${SCRIPT_NAME} debug ${target}"
      log "Run log file: ${RUN_LOG}"
      log "Run directory: ${LOG_DIR}"
      local service
      for service in $(for_each_service "${target}"); do
        set_service_context "${service}"
        debug_service
      done
      ;;
    *)
      usage
      [[ -n "${action}" ]] && exit 1
      ;;
  esac
}

main "$@"
