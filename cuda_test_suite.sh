#!/usr/bin/env bash
# CUDA test runner for energy_storms_cuda

set -u

# ---------------------------
# Defaults
# ---------------------------
BINARY=""
TESTS_DIR="test_files"
EXPECTED_DIR="mytests/expected"
RESULTS_DIR="mytests/results"
CONFIG_FILE="mytests/tests.cfg"
RECORD=0
TIMEOUT_SECS=0

# Colors
COLOR_RED='\e[31m'
COLOR_GRN='\e[32m'
COLOR_YEL='\e[33m'
COLOR_BLD_GRN='\e[1;32m'
COLOR_RESET='\e[0m'

# ---------------------------
# Helper functions
# ---------------------------
function usage() {
  cat <<EOF
Usage: $0 --binary ./energy_storms_cuda [options]

Options:
  --binary <path>       Path to the energy_storms_cuda binary (required)
  --tests-dir <dir>     Directory containing input test files (default: test_files)
  --expected-dir <dir>  Directory with expected outputs (default: expected)
  --results-dir <dir>   Where to write run outputs (default: results)
  --config <file>       Config file listing tests (default: tests.cfg)
  --record              Record current outputs into expected/ (overwrite!)
  --help                Show this message
EOF
}

function die() {
  echo -e "${COLOR_RED}ERROR: $*${COLOR_RESET}" >&2
  exit 1
}

function sanitize_name() {
  echo "$1" | sed -E 's#[/\\ ]+#_#g' | sed -E 's/[^A-Za-z0-9._-]/_/g'
}

function run_single() {
  local layer_size="$1"
  local testname="$2"
  shift 2
  local files=("$@")

  local runname="${layer_size}__${testname}"
  runname=$(sanitize_name "$runname")
  local outpath="${RESULTS_DIR}/${runname}.out"
  mkdir -p "${RESULTS_DIR}"

  # Build command
  local cmd=("${BINARY}" "${layer_size}")
  for f in "${files[@]}"; do
    cmd+=("${TESTS_DIR}/${f}")
  done

  # Run and capture output
  if [[ ${TIMEOUT_SECS} -gt 0 ]] && command -v timeout >/dev/null 2>&1; then
    timeout ${TIMEOUT_SECS}s "${cmd[@]}" > "${outpath}" 2>&1 || true
  else
    "${cmd[@]}" > "${outpath}" 2>&1 || true
  fi

  mkdir -p "${EXPECTED_DIR}"
  local expected_path="${EXPECTED_DIR}/${runname}.out"

  if [[ ${RECORD} -eq 1 ]]; then
    cp -f "${outpath}" "${expected_path}"
    printf "Recorded golden output -> %s\n" "${expected_path}"
    return 0
  fi

  if [[ ! -f "${expected_path}" ]]; then
    printf "[NO GOLDEN] %s -> expected file does not exist\n" "$runname"
    return 2
  fi

  # Compare Results
  local expected_result=$(grep '^Result:' "$expected_path" || echo "")
  local actual_result=$(grep '^Result:' "$outpath" || echo "")

  if [[ "$expected_result" == "$actual_result" ]]; then
    local time_new=$(grep '^Time:' "$outpath" | awk '{print $2}' || echo "-")
    local time_ref=$(grep '^Time:' "$expected_path" | awk '{print $2}' || echo "-")
    
    local dt="-"
    local dt_color="${COLOR_RESET}"

    if [[ "$time_new" != "-" && "$time_ref" != "-" ]]; then
      # Calculate delta
      dt=$(awk -v a="$time_new" -v b="$time_ref" 'BEGIN{printf("%.6f", (a-b))}')
      
      # Determine color: negative (faster) is green, positive (slower) is yellow
      local is_faster=$(awk -v d="$dt" 'BEGIN{print (d < 0 ? 1 : 0)}')
      local is_slower=$(awk -v d="$dt" 'BEGIN{print (d > 0 ? 1 : 0)}')
      
      if [[ "$is_faster" -eq 1 ]]; then
        dt_color="${COLOR_BLD_GRN}"
        dt="-$dt" # Format fix for display if needed
      elif [[ "$is_slower" -eq 1 ]]; then
        dt_color="${COLOR_YEL}"
        dt="+$dt"
      fi
    fi

    printf "${COLOR_GRN}PASS${COLOR_RESET} %s (time: %s ; ref: %s ; delta: %b%s${COLOR_RESET})\n" \
           "$runname" "$time_new" "$time_ref" "$dt_color" "$dt"
    return 0
  else
    printf "${COLOR_RED}FAIL${COLOR_RESET} %s (Result mismatch)\n" "$runname"
    diff -u <(echo "$expected_result") <(echo "$actual_result") || true
    return 1
  fi
}

# ---------------------------
# Parse args
# ---------------------------
if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BINARY="$2"; shift 2;;
    --tests-dir) TESTS_DIR="$2"; shift 2;;
    --expected-dir) EXPECTED_DIR="$2"; shift 2;;
    --results-dir) RESULTS_DIR="$2"; shift 2;;
    --config) CONFIG_FILE="$2"; shift 2;;
    --record) RECORD=1; shift 1;;
    --help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

if [[ -z "$BINARY" ]]; then die "--binary is required"; fi
if [[ ! -x "$BINARY" ]]; then die "Binary not found/executable: $BINARY"; fi

# Ensure config exists
if [[ ! -f "$CONFIG_FILE" ]]; then
  die "Config file $CONFIG_FILE not found."
fi

# ---------------------------
# Read config and Run
# ---------------------------
mapfile -t CONFIG_LINES < <(grep -v '^#' "$CONFIG_FILE" | sed '/^\s*$/d')

TOTAL=0; PASSED=0; FAILED=0; MISSING=0

for line in "${CONFIG_LINES[@]}"; do
    read -r -a tokens <<< "$line"
    testname="${tokens[0]}"
    layer_size="${tokens[1]}"
    files=("${tokens[@]:2}")

    # Check files exist
    skip=0
    for f in "${files[@]}"; do
      if [[ ! -f "${TESTS_DIR}/${f}" ]]; then
        printf "${COLOR_YEL}Missing input:${COLOR_RESET} %s\n" "$f"
        skip=1
      fi
    done
    [[ $skip -eq 1 ]] && { ((MISSING++)); continue; }

    ((TOTAL++))
    run_single "${layer_size}" "${testname}" "${files[@]}"
    rc=$?
    [[ $rc -eq 0 ]] && ((PASSED++))
    [[ $rc -eq 1 ]] && ((FAILED++))
    [[ $rc -eq 2 ]] && ((MISSING++))
done

printf "\nSummary: total=%d ${COLOR_GRN}passed=%d${COLOR_RESET} ${COLOR_RED}failed=%d${COLOR_RESET} missing=%d\n" \
       "$TOTAL" "$PASSED" "$FAILED" "$MISSING"

[[ $FAILED -gt 0 ]] && exit 2
exit 0
