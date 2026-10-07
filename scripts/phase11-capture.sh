#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s [duration_seconds] [sample_interval_seconds] [output_root] [workload_label]\n' "$0" >&2
}

duration=${1:-60}
interval=${2:-1}
output_root=${3:-/var/lib/orion/phase11}
workload_label=${4:-unspecified}
if [[ ! "$duration" =~ ^[1-9][0-9]*$ || ! "$interval" =~ ^[1-9][0-9]*$ ]]; then
    usage
    exit 2
fi

umask 027
run_id=$(date -u +%Y%m%dT%H%M%SZ)
out_dir="$output_root/$run_id"
install -d -o orion-service -g orion-service -m 0750 "$output_root"
install -d -o orion-service -g orion-service -m 0750 "$out_dir"
started_at=$(date --iso-8601=seconds)
started_epoch=$(date +%s)

{
    printf 'capture_id=%s\n' "$run_id"
    printf 'capture_started=%s\n' "$started_at"
    printf 'capture_duration_seconds=%s\n' "$duration"
    printf 'sample_interval_seconds=%s\n' "$interval"
    printf 'workload_label=%s\n' "$workload_label"
    printf 'cpu_percent_basis=aggregate process CPU across cores; may exceed 100\n'
    printf 'hostname=%s\n' "$(hostname)"
    printf 'device_model='
    if [[ -r /proc/device-tree/model ]]; then tr -d '\000' < /proc/device-tree/model; fi
    printf '\nos_release='; . /etc/os-release; printf '%s\n' "$PRETTY_NAME"
    printf 'kernel=%s\n' "$(uname -r)"
    printf 'architecture=%s\n' "$(uname -m)"
    printf 'cpu_model='
    awk -F': ' '/Model name|Hardware/ { print $2; exit }' /proc/cpuinfo
    printf 'cpu_cores=%s\n' "$(getconf _NPROCESSORS_ONLN)"
    printf 'memory_total_bytes=%s\n' "$(awk '/MemTotal:/ {print $2 * 1024}' /proc/meminfo)"
    printf 'python_voice=%s\n' "$(/opt/orion/.venv-voice/bin/python --version 2>&1)"
    printf 'uv_version=%s\n' "$(/usr/local/bin/uv --version 2>&1)"
    printf 'ollama_version=%s\n' "$(ollama --version 2>&1)"
    (cd /var/lib/orion && UV_NO_CONFIG=1 UV_CACHE_DIR=/var/cache/orion/uv-cache /usr/local/bin/uv pip show --python /opt/orion/.venv-voice/bin/python orion-assistant kittentts vosk openwakeword onnxruntime gpiozero lgpio) 2>&1
    printf '\nSYSTEMD\n'
    systemctl show orion.service -p ActiveState -p SubState -p User -p Group -p ExecStart -p MainPID
    systemctl show ollama.service -p ActiveState -p SubState -p MainPID
    printf '\nOLLAMA_MODEL\n'
    ollama list
    ollama show qwen3:1.7b 2>&1 || true
    ollama ps
    printf '\nSERVICE_CONFIG_NON_SECRET\n'
    grep -E '^(ORION_MODEL_CONTEXT_SIZE|ORION_MODEL_THREADS|ORION_MODEL_NAME|ORION_MODEL_BACKEND|ORION_AUDIO_SAMPLE_RATE|ORION_AUDIO_QUEUE_CHUNKS|ORION_VOICE_PROFILE|ORION_STT_BACKEND|ORION_STT_MODEL_PATH|ORION_WAKE_WORD_MODEL_PATH|ORION_WAKE_WORD_MELSPECTROGRAM_PATH|ORION_WAKE_WORD_EMBEDDING_PATH|ORION_WAKE_WORD_ENABLED)=' /etc/orion/orion.env || true
    printf '\nAUDIO_DEVICES\n'
    arecord -l 2>&1 || true
    aplay -l 2>&1 || true
    printf '\nPOWER_AND_COOLING\n'
    if command -v vcgencmd >/dev/null 2>&1; then
        vcgencmd measure_temp 2>&1 || true
        vcgencmd get_throttled 2>&1 || true
        vcgencmd get_config int 2>&1 | grep -E 'arm_freq|gpu_mem|over_voltage' || true
    else
        printf 'vcgencmd unavailable\n'
    fi
    printf 'power_supply_cooling_ambient=record manually; not reliably discoverable from software\n'
    printf '\nPACKAGE_CHECK\n'
    (cd /var/lib/orion && UV_NO_CONFIG=1 UV_CACHE_DIR=/var/cache/orion/uv-cache /usr/local/bin/uv pip check --python /opt/orion/.venv-voice/bin/python) 2>&1 || true
    printf '\nLOCK_HASH\n'
    sha256sum /opt/orion/uv.lock
} > "$out_dir/metadata.txt" 2>&1

printf 'timestamp_utc,elapsed_seconds,load_1m,mem_available_bytes,swap_used_bytes,cpu_temp,throttled,orion_cpu_percent,orion_rss_kb,ollama_cpu_percent,ollama_rss_kb\n' > "$out_dir/resources.csv"
clock_ticks=$(getconf CLK_TCK)
page_kb=$(( $(getconf PAGESIZE) / 1024 ))
process_totals() {
    local user=$1 pid rest
    local total_ticks=0 total_rss_kb=0
    local -a fields
    while read -r pid; do
        [[ -r "/proc/$pid/stat" ]] || continue
        rest=$(sed 's/^.*) //' "/proc/$pid/stat")
        read -r -a fields <<< "$rest"
        (( ${#fields[@]} > 21 )) || continue
        total_ticks=$((total_ticks + fields[11] + fields[12]))
        total_rss_kb=$((total_rss_kb + fields[21] * page_kb))
    done < <(ps -u "$user" -o pid= 2>/dev/null || true)
    printf '%s %s\n' "$total_ticks" "$total_rss_kb"
}
read -r previous_orion_ticks previous_orion_rss < <(process_totals orion-service)
read -r previous_ollama_ticks previous_ollama_rss < <(process_totals ollama)
previous_sample_epoch=$(date +%s)
while :; do
    sleep "$interval"
    now=$(date +%s)
    elapsed=$((now - started_epoch))
    if (( elapsed >= duration )); then
        break
    fi
    load_1m=$(awk '{print $1}' /proc/loadavg)
    mem_available=$(awk '/MemAvailable:/ {print $2 * 1024}' /proc/meminfo)
    swap_used=$(awk '/SwapTotal:/ {total=$2} /SwapFree:/ {free=$2} END {print (total-free) * 1024}' /proc/meminfo)
    if command -v vcgencmd >/dev/null 2>&1; then
        cpu_temp=$(vcgencmd measure_temp 2>/dev/null | sed -n "s/^temp=\([0-9.]*\).*/\1/p")
        throttled=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)
    else
        cpu_temp=
        throttled=
    fi
    read -r orion_ticks orion_rss < <(process_totals orion-service)
    read -r ollama_ticks ollama_rss < <(process_totals ollama)
    sample_seconds=$((now - previous_sample_epoch))
    (( sample_seconds > 0 )) || sample_seconds=1
    orion_cpu=$(awk -v delta="$((orion_ticks - previous_orion_ticks))" -v hz="$clock_ticks" -v seconds="$sample_seconds" 'BEGIN {printf "%.1f", delta * 100 / (hz * seconds)}')
    ollama_cpu=$(awk -v delta="$((ollama_ticks - previous_ollama_ticks))" -v hz="$clock_ticks" -v seconds="$sample_seconds" 'BEGIN {printf "%.1f", delta * 100 / (hz * seconds)}')
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$elapsed" "$load_1m" \
        "$mem_available" "$swap_used" "$cpu_temp" "$throttled" \
        "$orion_cpu" "$orion_rss" "$ollama_cpu" "$ollama_rss" \
        >> "$out_dir/resources.csv"
    previous_orion_ticks=$orion_ticks
    previous_orion_rss=$orion_rss
    previous_ollama_ticks=$ollama_ticks
    previous_ollama_rss=$ollama_rss
    previous_sample_epoch=$now
done

journalctl -u orion.service --since "$started_at" --no-pager > "$out_dir/orion-journal.log" 2>&1 || \
    printf 'Could not read system journal; run this capture with sudo to include service logs.\n' > "$out_dir/orion-journal.log"

awk -F, 'NR > 1 {if ($9 > max_orion_rss) max_orion_rss=$9; if ($11 > max_ollama_rss) max_ollama_rss=$11; if (NR == 2 || $4 < min_mem) min_mem=$4; if ($5 > max_swap) max_swap=$5; if ($6 != "" && $6 > max_temp) max_temp=$6; if ($8 > max_orion_cpu) max_orion_cpu=$8; if ($10 > max_ollama_cpu) max_ollama_cpu=$10} END {printf "samples=%d\nmax_orion_rss_kb=%.0f\nmax_ollama_rss_kb=%.0f\nmin_mem_available_bytes=%.0f\nmax_swap_used_bytes=%.0f\nmax_cpu_temp_c=%.1f\nmax_orion_cpu_percent=%.1f\nmax_ollama_cpu_percent=%.1f\n", NR-1, max_orion_rss, max_ollama_rss, min_mem, max_swap, max_temp, max_orion_cpu, max_ollama_cpu}' \
    "$out_dir/resources.csv" > "$out_dir/summary.txt"
printf 'Capture complete: %s\n' "$out_dir"
chown -R orion-service:orion-service "$out_dir"
