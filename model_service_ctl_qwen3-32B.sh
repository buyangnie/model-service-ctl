#!/usr/bin/env bash
set -euo pipefail

#######################################
# 通用 vLLM 容器管理脚本（单服务版）
# 当前默认环境：
#   - 脚本目录: /opsfactory/model_service
#   - 模型目录: /data/models/Qwen3-32B
#
# Notes:
#   - Qwen3-32B 默认开启 thinking。
#   - 若线上追求更低 TTFT，建议在客户端 prompt 或 system message 中显式附加 `/no_think`。
#   - vLLM 官方 Qwen tool calling 建议使用 `--tool-call-parser hermes`。
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
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-x86_64}"
API_KEY="${API_KEY:-change-me}"

MODELS_ROOT="${MODELS_ROOT:-/data/models}"
HF_CACHE="${HF_CACHE:-/data/hf_cache}"

CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen3-32b}"
MODEL_DIR="${MODEL_DIR:-/data/models/Qwen3-32B}"
MODEL_NAME="${MODEL_NAME:-Qwen3-32B}"

HOST_PORT="${HOST_PORT:-8000}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"

GPU_DEVICES="${GPU_DEVICES:-device=0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"

DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

# vLLM 官方 Qwen tool calling 建议：
#   --tool-call-parser hermes
# Goose 这类客户端通常会发送 tool_choice=auto，因此默认启用 auto tool choice。
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
# Qwen3-32B 的 thinking 切换主要发生在请求侧，而不是 serve CLI。
# 这里保留一个显式变量，便于测试时统一记录当前预期模式：
#   think      -> 期望客户端请求时显式附加 /think
#   no_think   -> 期望客户端请求时显式附加 /no_think
THINKING_MODE="${THINKING_MODE:-think}"
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
  ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE}
  LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY}
  THINKING_MODE=${THINKING_MODE}
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
# Diagnostics
#######################################
dump_env_snapshot() {
  resolve_model_dir
  local outfile="${LOG_DIR}/env_snapshot.txt"
  {
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
    echo "enable_auto_tool_choice=${ENABLE_AUTO_TOOL_CHOICE}"
    echo "language_model_only=${LANGUAGE_MODEL_ONLY}"
    echo "thinking_mode=${THINKING_MODE}"
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
  } >"${outfile}" 2>&1 || true
}

#######################################
# Validation
#######################################
validate_env() {
  resolve_model_dir
  divider
  log "Validating runtime environment"

  require_bin "${DOCKER_BIN}"
  require_bin "${CURL_BIN}"

  [[ -d "${MODELS_ROOT}" ]] || { err "Models root not found: ${MODELS_ROOT}"; return 1; }
  [[ -d "${MODEL_DIR}" ]] || { err "Model dir not found: ${MODEL_DIR}"; return 1; }
  [[ -d "${MODEL_DIR_REAL}" ]] || { err "Resolved model dir not found: ${MODEL_DIR_REAL}"; return 1; }

  case "${THINKING_MODE}" in
    think|no_think) ;;
    *)
      err "Invalid THINKING_MODE: ${THINKING_MODE}. Expected: think | no_think"
      return 1
      ;;
  esac

  if [[ ! -f "${MODEL_DIR_REAL}/config.json" && ! -f "${MODEL_DIR_REAL}/params.json" ]]; then
    err "MODEL_DIR does not look like a valid model root: ${MODEL_DIR_REAL}"
    return 1
  fi

  mkdir -p "${HF_CACHE}"

  log "Validation passed:"
  log "  MODELS_ROOT = ${MODELS_ROOT}"
  log "  MODEL_DIR   = ${MODEL_DIR}"
  log "  MODEL_DIR_REAL = ${MODEL_DIR_REAL}"
  log "  HF_CACHE    = ${HF_CACHE}"
  log "  HOST_PORT   = ${HOST_PORT}"
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
  local cmd_file="${LOG_DIR}/start_command.txt"
  mkdir -p "${outdir}"

  warn "Collecting failure context for ${CONTAINER_NAME} into ${outdir}"
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
    echo "model_extra_mount=$(printf '%s' "${MODEL_EXTRA_MOUNT}")"
    echo

    echo "==================== start_command.txt ===================="
    cat "${cmd_file}" 2>/dev/null || true
    echo
  } >"${outdir}/summary.txt"

  warn "Failure context ready: ${outdir}"
}

health_check() {
  local retries="${HEALTH_RETRIES}"
  local interval="${HEALTH_INTERVAL}"
  local attempt=1

  log "Checking health for ${CONTAINER_NAME} on port ${HOST_PORT} (timeout=$((retries * interval))s)"
  while (( attempt <= retries )); do
    if container_seems_healthy; then
      log "${CONTAINER_NAME} is ready"
      return 0
    fi

    local status
    status="$(${DOCKER_BIN} inspect --format '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo unknown)"
    if [[ "${status}" != "running" ]]; then
      err "${CONTAINER_NAME} is not running during health check (status=${status})"
      collect_failure_context "container_not_running"
      return 1
    fi

    if (( attempt % 10 == 0 )); then
      log "Health check attempt ${attempt}/${retries} not ready yet"
    fi
    sleep "${interval}"
    attempt=$((attempt + 1))
  done

  err "${CONTAINER_NAME} health check failed after ${retries} attempts"
  collect_failure_context "health_timeout"
  return 1
}

debug_service() {
  resolve_model_dir
  local outdir="${LOG_DIR}/debug_${CONTAINER_NAME}"
  mkdir -p "${outdir}"
  log "Collecting debug context into ${outdir}"
  capture_cmd "${outdir}/docker_ps.txt" ${DOCKER_BIN} ps -a
  capture_cmd "${outdir}/docker_inspect.txt" ${DOCKER_BIN} inspect "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_inspect_state.txt" ${DOCKER_BIN} inspect --format '{{.State.Status}} restart_count={{.RestartCount}} exit_code={{.State.ExitCode}} oom_killed={{.State.OOMKilled}} error={{.State.Error}}' "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_logs_tail_200.txt" ${DOCKER_BIN} logs --timestamps --tail 200 "${CONTAINER_NAME}"
  capture_cmd "${outdir}/docker_logs_tail_500.txt" ${DOCKER_BIN} logs --timestamps --tail 500 "${CONTAINER_NAME}"
  capture_cmd "${outdir}/nvidia_smi.txt" bash -lc "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true"
  capture_cmd "${outdir}/port_check.txt" port_check "${HOST_PORT}"
  capture_cmd "${outdir}/model_dir_ls.txt" ls -lah "${MODEL_DIR}"

  {
    echo "container_name=${CONTAINER_NAME}"
    echo "model_dir=${MODEL_DIR}"
    echo "model_name=${MODEL_NAME}"
    echo "host_port=${HOST_PORT}"
    echo "gpu_devices=${GPU_DEVICES}"
  } >"${outdir}/summary.txt"

  log "Debug bundle written to ${outdir}"
}

#######################################
# Lifecycle
#######################################
start_service() {
  divider
  log "Script start: ${SCRIPT_NAME} startup"
  log "Run log file: ${RUN_LOG}"
  log "Run directory: ${LOG_DIR}"

  dump_env_snapshot
  validate_env

  if is_port_listening "${HOST_PORT}"; then
    warn "Host port ${HOST_PORT} already appears to be listening"
    port_check "${HOST_PORT}" | tee -a "${RUN_LOG}" || true
  fi

  remove_container_if_exists

  local cmd_str
  cmd_str="$(build_run_cmd)"
  echo "${cmd_str}" >"${LOG_DIR}/start_command.txt"

  log "Starting container: ${CONTAINER_NAME}"
  if ! bash -lc "${cmd_str}" >>"${RUN_LOG}" 2>&1; then
    err "docker run failed"
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
  echo "ENABLE_AUTO_TOOL_CHOICE : ${ENABLE_AUTO_TOOL_CHOICE}" | tee -a "${RUN_LOG}"
  echo "CHAT_TEMPLATE : ${CHAT_TEMPLATE}" | tee -a "${RUN_LOG}"
  echo "LANGUAGE_MODEL_ONLY : ${LANGUAGE_MODEL_ONLY}" | tee -a "${RUN_LOG}"
  echo "THINKING_MODE : ${THINKING_MODE}" | tee -a "${RUN_LOG}"
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
  echo "Health Probe" | tee -a "${RUN_LOG}"
  divider
  if container_seems_healthy; then
    echo "READY" | tee -a "${RUN_LOG}"
  else
    echo "NOT_READY" | tee -a "${RUN_LOG}"
  fi
}

#######################################
# Main
#######################################
main() {
  local action="${1:-}"
  case "${action}" in
    validate)
      log "Script start: ${SCRIPT_NAME} validate"
      log "Run log file: ${RUN_LOG}"
      log "Run directory: ${LOG_DIR}"
      dump_env_snapshot
      validate_env
      ;;
    startup)
      start_service
      ;;
    stop)
      log "Script start: ${SCRIPT_NAME} stop"
      log "Run log file: ${RUN_LOG}"
      log "Run directory: ${LOG_DIR}"
      stop_service
      ;;
    restart)
      restart_service
      ;;
    status)
      log "Script start: ${SCRIPT_NAME} status"
      log "Run log file: ${RUN_LOG}"
      log "Run directory: ${LOG_DIR}"
      status_service
      ;;
    logs)
      log "Script start: ${SCRIPT_NAME} logs"
      log "Run log file: ${RUN_LOG}"
      log "Run directory: ${LOG_DIR}"
      logs_service
      ;;
    debug)
      log "Script start: ${SCRIPT_NAME} debug"
      log "Run log file: ${RUN_LOG}"
      log "Run directory: ${LOG_DIR}"
      debug_service
      ;;
    *)
      usage
      [[ -n "${action}" ]] && exit 1
      ;;
  esac
}

main "$@"
