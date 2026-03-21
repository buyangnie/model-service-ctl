#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
RUN_TS="$(date '+%Y-%m-%d %H:%M:%S')"
HOSTNAME="$(hostname 2>/dev/null || echo unknown)"
OS_NAME="$(uname -s 2>/dev/null || echo unknown)"

section() {
  printf '\n==================== %s ====================\n' "$1"
}

print_kv() {
  local key="$1"
  local value="${2:-}"
  printf '%-24s %s\n' "${key}" "${value}"
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

safe_run() {
  if "$@" 2>/dev/null; then
    return 0
  fi
  return 1
}

print_top_mem_processes() {
  if ! has_cmd ps; then
    echo "ps command not found"
    return 0
  fi

  if ps -eo pid,ppid,user,%cpu,%mem,comm --sort=-%mem >/dev/null 2>&1; then
    ps -eo pid,ppid,user,%cpu,%mem,comm --sort=-%mem | head -n 15
  else
    ps aux | head -n 15
  fi
}

print_top_cpu_processes() {
  if ! has_cmd ps; then
    echo "ps command not found"
    return 0
  fi

  if ps -eo pid,ppid,user,%cpu,%mem,comm --sort=-%cpu >/dev/null 2>&1; then
    ps -eo pid,ppid,user,%cpu,%mem,comm --sort=-%cpu | head -n 15
  else
    ps aux | head -n 15
  fi
}

get_os_pretty_name() {
  if [[ -r /etc/os-release ]]; then
    awk -F= '/^PRETTY_NAME=/{gsub(/^"|"$/, "", $2); print $2}' /etc/os-release
    return 0
  fi
  return 1
}

get_cpu_model() {
  if [[ -r /proc/cpuinfo ]]; then
    awk -F: '/model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo
    return 0
  fi
  return 1
}

get_cpu_socket_count() {
  if [[ -r /proc/cpuinfo ]]; then
    awk -F: '/physical id/ {gsub(/^[ \t]+/, "", $2); print $2}' /proc/cpuinfo | sort -u | wc -l | awk '{print $1}'
    return 0
  fi
  return 1
}

get_default_ip() {
  if has_cmd ip; then
    ip route get 1.1.1.1 2>/dev/null | awk '/src/ {for (i=1; i<=NF; i++) if ($i == "src") {print $(i+1); exit}}'
    return 0
  fi
  if has_cmd ifconfig; then
    ifconfig 2>/dev/null | awk '
      /^[a-z0-9]/ {iface=$1; sub(/:$/, "", iface)}
      /inet / && $2 != "127.0.0.1" {print $2; exit}
    '
    return 0
  fi
  return 1
}

section "META"
print_kv "script" "${SCRIPT_NAME}"
print_kv "generated_at" "${RUN_TS}"
print_kv "hostname" "${HOSTNAME}"
print_kv "user" "$(whoami 2>/dev/null || echo unknown)"
print_kv "uptime" "$(uptime -p 2>/dev/null || uptime 2>/dev/null || echo unavailable)"
print_kv "os_family" "${OS_NAME}"

section "OS"
print_kv "pretty_name" "$(get_os_pretty_name || echo unavailable)"
print_kv "kernel" "$(uname -srmo 2>/dev/null || uname -a 2>/dev/null || echo unavailable)"
print_kv "architecture" "$(uname -m 2>/dev/null || echo unavailable)"
print_kv "virtualization" "$(systemd-detect-virt 2>/dev/null || echo none)"

section "CPU"
print_kv "model" "$(get_cpu_model || echo unavailable)"
print_kv "logical_cpus" "$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo unavailable)"
if has_cmd lscpu; then
  print_kv "sockets" "$(lscpu 2>/dev/null | awk -F: '/Socket\(s\)/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
  print_kv "cores_per_socket" "$(lscpu 2>/dev/null | awk -F: '/Core\(s\) per socket/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
  print_kv "threads_per_core" "$(lscpu 2>/dev/null | awk -F: '/Thread\(s\) per core/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
else
  print_kv "sockets" "$(get_cpu_socket_count || echo unavailable)"
fi
print_kv "load_average" "$(awk '{print $1" "$2" "$3}' /proc/loadavg 2>/dev/null || uptime 2>/dev/null | sed -E 's/.*load averages?: //' || echo unavailable)"

section "MEMORY"
if has_cmd free; then
  free -h
elif [[ "${OS_NAME}" == "Darwin" ]] && has_cmd vm_stat; then
  vm_stat
  echo
  sysctl hw.memsize 2>/dev/null || true
else
  echo "free command not found"
fi

section "DISK"
if has_cmd df; then
  if df -hT >/dev/null 2>&1; then
    df -hT
  else
    df -h
  fi
else
  echo "df command not found"
fi

section "BLOCK_DEVICES"
if has_cmd lsblk; then
  lsblk -o NAME,FSTYPE,SIZE,TYPE,MOUNTPOINT,MODEL
else
  echo "lsblk command not found"
fi

section "NETWORK_INTERFACES"
if has_cmd ip; then
  ip -brief addr
elif has_cmd ifconfig; then
  ifconfig
else
  echo "ip/ifconfig not found"
fi

section "PRIMARY_NETWORK"
print_kv "default_ip" "$(get_default_ip || echo unavailable)"
if has_cmd ip; then
  print_kv "default_route" "$(ip route show default 2>/dev/null | head -n 1 || echo unavailable)"
else
  print_kv "default_route" "unavailable"
fi

section "DNS"
if [[ -r /etc/resolv.conf ]]; then
  cat /etc/resolv.conf
else
  echo "/etc/resolv.conf not readable"
fi

section "TIME"
print_kv "date" "$(date 2>/dev/null || echo unavailable)"
if has_cmd timedatectl; then
  timedatectl 2>/dev/null || true
else
  echo "timedatectl not found"
fi

section "TOP_MEMORY_PROCESSES"
print_top_mem_processes

section "TOP_CPU_PROCESSES"
print_top_cpu_processes

section "NVIDIA_SMI"
if has_cmd nvidia-smi; then
  nvidia-smi
  echo
  nvidia-smi -L || true
else
  echo "nvidia-smi not found"
fi

section "DOCKER"
if has_cmd docker; then
  docker version 2>/dev/null || true
  echo
  docker info 2>/dev/null || true
else
  echo "docker not found"
fi
