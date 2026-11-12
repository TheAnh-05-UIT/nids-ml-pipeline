#!/usr/bin/env bash
# Map one local PCAP/CSV pair on an ARM64 node; preflight is the default.
# Mapping một cặp PCAP/CSV tại node ARM64; chỉ tạo index khi có --execute.
set -euo pipefail

usage() {
  echo 'Usage: bash scripts/node_mapping.sh --case CASE --output artifacts/runs/NEW_DIR [--execute]'
  echo 'Default: read-only preflight. No Docker, SSH, installation, training or cleanup.'
}

CASE=''
OUTPUT=''
EXECUTE=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --case|--output)
      if [ "$#" -lt 2 ]; then usage; exit 2; fi
      if [ "$1" = --case ]; then CASE="$2"; else OUTPUT="$2"; fi
      shift 2 ;;
    --execute) EXECUTE=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
if [ -z "$CASE" ] || [ -z "$OUTPUT" ]; then usage; exit 2; fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != aarch64 ]; then
  echo 'STOP: node launcher requires Linux aarch64; use the Python CLI for Windows validation.' >&2
  exit 2
fi
python3 -B -c 'import sys; assert sys.version_info >= (3,11), "Python 3.11+ required"'
TSHARK="$(command -v tshark)"
TSHARK="$(readlink -f "$TSHARK")"

# Accepted Ubuntu ARM64 TShark 4.2.2 identity from the operator's native preflight.
# Hash đã chốt của TShark ARM64; mismatch phải review, không tự nhận binary mới.
EXPECTED_TSHARK_SHA='e26bb4e95e7cc0297ee31dc0b78f89290c1fd4bcd1fa6d038a5821d7f49105de'
ARGS=(--config configs/multi-attack-pcap-v1.json --case "$CASE"
      --tshark "$TSHARK" --tshark-sha256 "$EXPECTED_TSHARK_SHA"
      --output "$OUTPUT")
if [ "$EXECUTE" = false ]; then ARGS+=(--dry-run); fi
exec python3 -B scripts/audit_full_pair.py "${ARGS[@]}"
