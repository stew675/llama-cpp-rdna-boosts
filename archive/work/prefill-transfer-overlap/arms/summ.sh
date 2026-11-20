#!/usr/bin/env bash
# Re-print the arm summary from an existing log (see arm.sh).  Usage: ./summ.sh runs/*.log
set -uo pipefail
for LOG in "$@"; do
  NAME=$(basename "$LOG" .log)
  pe=$(grep -oE "prompt eval time = *[0-9.]+ ms / *[0-9]+ tokens" "$LOG" | tail -1)
  ms=$(echo "$pe" | grep -oE "= *[0-9.]+" | grep -oE "[0-9.]+")
  ntok=$(echo "$pe" | grep -oE "/ *[0-9]+ tokens" | grep -oE "[0-9]+")
  fill=$(sed -nE 's/.*fills=([0-9.]+) MiB.*/\1/p' "$LOG" | tail -1)
  cold=$(sed -nE 's/.*cold-reads=([0-9.]+) MiB.*/\1/p' "$LOG" | tail -1)
  res=$(grep -oE "\([0-9.]+% residency\)" "$LOG" | head -1 | tr -d '()')
  short=$(grep -c "H2D staging shortfall" "$LOG" 2>/dev/null || true)
  # the H2D rate this run calibrated (varies per device count / slot)
  rate=$(sed -nE 's/.*H2D bandwidth calibration: ([0-9.]+) GB\/s.*/\1/p' "$LOG" | tail -1)
  tps=$(awk -v ms="${ms:-0}" -v n="${ntok:-0}" 'BEGIN { if (ms > 0) printf "%.1f", n / ms * 1000 }')
  printf '%-24s tps=%-8s ntok=%-6s prefill_ms=%-9s fill_MiB=%-8s cold=%-6s res=%-9s rate=%-5s short=%s\n' \
    "$NAME" "${tps:-?}" "${ntok:-?}" "${ms:-?}" "${fill:-None}" "${cold:-0}" "${res:-?}" "${rate:-?}" "${short:-0}"
  awk -v ms="${ms:-0}" -v fill="${fill:-0}" -v cold="${cold:-0}" -v rate="${rate:-28.7}" -v name="$NAME" 'BEGIN {
    if (ms <= 0) exit;
    dma_s = (fill + cold) * 1.048576 / (rate * 1000.0);
    printf "%-24s prefill=%.3fs dma=%.4fs F=%.4f\n", name, ms / 1000.0, dma_s, dma_s / (ms / 1000.0)
  }'
done
