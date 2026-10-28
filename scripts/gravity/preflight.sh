#!/usr/bin/env bash
# Source this file from cess.pbs, or run it with --help / --dry-run.
set -Eeuo pipefail

cess_fail() { printf 'CESS preflight: %s\n' "$*" >&2; exit 2; }

cess_prepare() {
    local name value flag
    for name in CESS_PYTHON CESS_REPO CESS_CONFIG CESS_INPUT CESS_OUTPUT CESS_MANIFEST; do
        value="${!name:-}"
        [[ "$value" = /* ]] || cess_fail "$name must be an explicit absolute path"
        export "$name"
    done
    [[ -x "$CESS_PYTHON" ]] || cess_fail 'CESS_PYTHON is not executable'
    [[ -d "$CESS_REPO/src/cess" ]] || cess_fail 'CESS_REPO does not contain src/cess'
    [[ -f "$CESS_CONFIG" && -r "$CESS_CONFIG" ]] || cess_fail 'Cannot read CESS_CONFIG'
    [[ -r "$CESS_INPUT" && ( -f "$CESS_INPUT" || ( -d "$CESS_INPUT" && -x "$CESS_INPUT" ) ) ]] || \
        cess_fail 'CESS_INPUT must be a readable HDF5 file or searchable input directory'
    [[ ! -e "$CESS_OUTPUT" || -d "$CESS_OUTPUT" ]] || cess_fail 'CESS_OUTPUT is not a directory'

    export PYTHONPATH="$CESS_REPO/src" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
    export CESS_EXPECTED_VERSION="${CESS_EXPECTED_VERSION:-0.9.0}"
    export CESS_EXPECTED_REVISION="${CESS_EXPECTED_REVISION:-20261027-D5}"
    export CESS_WORKERS="${CESS_WORKERS:-1}" CESS_CPUS="${CESS_CPUS:-1}"
    export CESS_CHUNK_SIZE="${CESS_CHUNK_SIZE:-}" CESS_SEED="${CESS_SEED:-1214}"
    export CESS_MEMORY_BUDGET_GIB="${CESS_MEMORY_BUDGET_GIB:-}"
    export CESS_SOURCE_START="${CESS_SOURCE_START:-0}" CESS_SOURCE_STOP="${CESS_SOURCE_STOP:-}"
    export CESS_MODE="${CESS_MODE:-standard}" CESS_OUTPUT_LAYOUT="${CESS_OUTPUT_LAYOUT:-structured}"
    export CESS_RESUME="${CESS_RESUME:-0}"
    export CESS_MORPHOLOGY="${CESS_MORPHOLOGY:-0}" CESS_WAVELENGTH_ERROR="${CESS_WAVELENGTH_ERROR:-0}"
    export CESS_PHOTOMETRY="${CESS_PHOTOMETRY:-0}" CESS_EMISSION_LINES="${CESS_EMISSION_LINES:-0}"

    "$CESS_PYTHON" -s -B - <<'PY'
import json
import math
import os
from pathlib import Path
import sys

def require(condition, message):
    if not condition:
        raise SystemExit('CESS preflight: ' + message)

require(sys.version_info[:2] == (3, 12), 'Python 3.12 is required')
import cess
from cess.contracts import REPAIR_REVISION
from cess.config.loader import load_config
require(Path(cess.__file__).resolve().parent == Path(os.environ['CESS_REPO']).resolve() / 'src/cess',
        'Imported package does not belong to CESS_REPO')
require(cess.__version__ == os.environ['CESS_EXPECTED_VERSION'], 'Package version mismatch')
require(REPAIR_REVISION == os.environ['CESS_EXPECTED_REVISION'], 'Repair revision mismatch')
for key in ('CESS_WORKERS', 'CESS_CPUS', 'CESS_SEED', 'CESS_SOURCE_START'):
    value = os.environ[key]
    require(value.isascii() and value.isdecimal(), key + ' must be a nonnegative integer')
    require(int(value) > 0 or key in ('CESS_SEED', 'CESS_SOURCE_START'), key + ' must be positive')
require(int(os.environ['CESS_WORKERS']) <= int(os.environ['CESS_CPUS']),
        'Workers exceed CESS_CPUS; request scheduler resources before raising the declared CPU limit')
chunk_size = os.environ['CESS_CHUNK_SIZE']
if chunk_size:
    require(chunk_size.isascii() and chunk_size.isdecimal() and int(chunk_size) > 0,
            'CESS_CHUNK_SIZE must be empty (whole-file tasks) or a positive integer')
memory_budget = os.environ['CESS_MEMORY_BUDGET_GIB']
if memory_budget:
    try:
        budget = float(memory_budget)
    except ValueError:
        raise SystemExit('CESS preflight: CESS_MEMORY_BUDGET_GIB must be positive and finite')
    require(math.isfinite(budget) and budget > 0,
            'CESS_MEMORY_BUDGET_GIB must be positive and finite')
stop = os.environ['CESS_SOURCE_STOP']
if stop:
    require(stop.isascii() and stop.isdecimal() and int(stop) >= int(os.environ['CESS_SOURCE_START']),
            'CESS_SOURCE_STOP must be an integer at least CESS_SOURCE_START')
for key in ('CESS_RESUME', 'CESS_MORPHOLOGY', 'CESS_WAVELENGTH_ERROR', 'CESS_PHOTOMETRY', 'CESS_EMISSION_LINES'):
    require(os.environ[key] in ('0', '1'), key + ' must be 0 or 1')
require(os.environ['CESS_MODE'] in ('standard', 'paired-wce'), 'Invalid CESS_MODE')
require(os.environ['CESS_OUTPUT_LAYOUT'] in ('structured', 'legacy'), 'Invalid CESS_OUTPUT_LAYOUT')
manifest = Path(os.environ['CESS_MANIFEST'])
require(manifest.is_file() if os.environ['CESS_RESUME'] == '1' else not (manifest.exists() or manifest.is_symlink()),
        'Resume requires an existing manifest; a new run requires a fresh manifest path')
require(manifest.resolve() not in (Path(os.environ['CESS_INPUT']).resolve(), Path(os.environ['CESS_CONFIG']).resolve()),
        'Manifest must not be the input or config file')
config = load_config(os.environ['CESS_CONFIG'])
runtime = {'input_path', 'output_path', 'el_input_path', 'el_output_path'}
for key, value in config.items():
    if not key.startswith('_') and key not in runtime and key.endswith(('_path', '_pkl')):
        require(Path(value).is_file() and os.access(value, os.R_OK), 'Cannot read resource ' + key + ': ' + str(value))
print(json.dumps({'python': sys.executable, 'version': cess.__version__, 'revision': REPAIR_REVISION,
                  'package': str(Path(cess.__file__).resolve()), 'config': os.environ['CESS_CONFIG'],
                  'declared_cpus': int(os.environ['CESS_CPUS']), 'workers': int(os.environ['CESS_WORKERS']),
                  'threads_per_worker': 1,
                  'input_kind': 'directory' if Path(os.environ['CESS_INPUT']).is_dir() else 'file',
                  'chunk_size': int(chunk_size) if chunk_size else None,
                  'memory_budget_gib': budget if memory_budget else None}, indent=2))
PY

    CESS_COMMAND=("$CESS_PYTHON" -s -B -m cess simulate
        --config "$CESS_CONFIG" --input "$CESS_INPUT" --output "$CESS_OUTPUT"
        --output-layout "$CESS_OUTPUT_LAYOUT" --mode "$CESS_MODE"
        --workers "$CESS_WORKERS" --threads-per-worker 1 --seed "$CESS_SEED"
        --source-start "$CESS_SOURCE_START")
    [[ -z "$CESS_CHUNK_SIZE" ]] || CESS_COMMAND+=(--chunk-size "$CESS_CHUNK_SIZE")
    [[ -z "$CESS_MEMORY_BUDGET_GIB" ]] || CESS_COMMAND+=(--memory-budget-gib "$CESS_MEMORY_BUDGET_GIB")
    [[ -z "$CESS_SOURCE_STOP" ]] || CESS_COMMAND+=(--source-stop "$CESS_SOURCE_STOP")
    for name in MORPHOLOGY WAVELENGTH_ERROR PHOTOMETRY EMISSION_LINES; do
        flag="CESS_$name"
        if [[ "${!flag}" == 1 ]]; then
            case "$name" in
                MORPHOLOGY) value=--morphology ;;
                WAVELENGTH_ERROR) value=--wavelength-error ;;
                PHOTOMETRY) value=--photometry ;;
                EMISSION_LINES) value=--emission-lines ;;
            esac
            CESS_COMMAND+=("$value")
        fi
    done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:---dry-run}" in
        --help|-h)
            printf '%s\n' 'Usage: bash preflight.sh [--dry-run|--help]' \
                'Requires absolute CESS_PYTHON, CESS_REPO, CESS_CONFIG, CESS_INPUT, CESS_OUTPUT, CESS_MANIFEST.' \
                'Checks version/resources; previews an input HDF5 file or directory without simulation products.' \
                'CESS_CHUNK_SIZE empty/unset keeps one task/output per input file; a positive value enables chunks.' \
                'New-run preview does not write a manifest; resume preview verifies the existing manifest read-only.' \
                'See scripts/gravity/README.md for switches, PBS resources, and submission.'
            ;;
        --dry-run)
            [[ $# -le 1 ]] || cess_fail 'Unexpected arguments'
            cess_prepare
            # Match the PBS payload's cwd: bundled runtime-path defaults are
            # part of the strict resume snapshot even with explicit input/output.
            cd -- "$CESS_REPO"
            if [[ "$CESS_RESUME" == 1 ]]; then
                CESS_COMMAND+=(--manifest "$CESS_MANIFEST" --resume)
            fi
            printf 'Preview command:'; printf ' %q' "${CESS_COMMAND[@]}" --dry-run; printf '\n'
            "${CESS_COMMAND[@]}" --dry-run
            ;;
        *) cess_fail 'Use --help or --dry-run' ;;
    esac
fi
