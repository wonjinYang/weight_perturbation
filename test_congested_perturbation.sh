#!/usr/bin/env bash

set -euo pipefail

# -------------------------------
# Colors and formatting
# -------------------------------
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
MAGENTA="\033[0;35m"
NC="\033[0m" # No Color

print_status() {
  local msg="$1"
  local color="${2:-$NC}"
  echo -e "${color}${msg}${NC}"
}

print_debug() {
  local msg="$1"
  if [[ "$VERBOSE" == "true" ]]; then
    print_status "[DEBUG] $msg" $CYAN
  fi
}

# -------------------------------
# Global variables
# -------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/test_results/logs"
PLOT_DIR="${ROOT_DIR}/test_results/plots"
SUMMARY_FILE="${ROOT_DIR}/test_results/test_summary.txt"
FAILED_LIST_FILE="${ROOT_DIR}/test_results/test_summary_failed.txt"

START_TS=$(date +%s)

# Global counters and results list
declare -i TOTAL_TESTS=0
declare -i PASSED=0
declare -i FAILED=0
declare -a TEST_RESULTS=()

# Configuration flags
VERBOSE=false
DRY_RUN=false
FAST_MODE=false
SKIP_DEPS=false

# Test selection flags
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_CONGESTION=false
RUN_COMPREHENSIVE=false
RUN_EXAMPLES=false
RUN_EXAMPLES_CONGESTION=false
RUN_PERFORMANCE=false
RUN_MEMORY=false
RUN_ALL=true

# Exclusion list
declare -a EXCLUDE_TESTS=()

# Individual test selection
declare -a INDIVIDUAL_TESTS=()

# -------------------------------
# Available test definitions
# -------------------------------
declare -A TEST_CATEGORIES=(
  ["unit"]="Unit Tests"
  ["integration"]="Integration Tests"
  ["congestion"]="Congestion Theory Tests"
  ["comprehensive"]="Comprehensive Examples"
  ["examples"]="Basic Examples"
  ["examples-congestion"]="Congestion Examples with Visualization"
  ["performance"]="Performance Tests"
  ["memory"]="Memory Tests"
)

declare -A UNIT_TESTS=(
  ["models"]="pytest -q src/tests/test_models.py"
  ["samplers"]="pytest -q src/tests/test_samplers.py"
  ["losses"]="pytest -q src/tests/test_losses.py"
  ["perturbation"]="pytest -q src/tests/test_perturbation.py"
  ["pretrain"]="pytest -q src/tests/test_pretrain.py"
)

declare -A INTEGRATION_TESTS=(
  ["integration"]="python3 src/tests/test_integration.py"
)

declare -A CONGESTION_TESTS=(
  ["congestion"]="python3 src/tests/test_congestion.py"
)

declare -A COMPREHENSIVE_TESTS=(
  ["comprehensive"]="comprehensive_test_special"
)

declare -A EXAMPLE_TESTS=(
  ["section2"]="python3 src/examples/example_section2.py"
  ["section3"]="python3 src/examples/example_section3.py"
)

declare -A EXAMPLE_CONGESTION_TESTS=(
  ["section2-congestion"]="python3 src/examples/example_with_congestion_section2.py"
  ["section3-congestion"]="python3 src/examples/example_with_congestion_section3.py"
)

declare -A PERFORMANCE_TESTS=(
  ["performance"]="python3 src/tests/test_performance.py"
)

declare -A MEMORY_TESTS=(
  ["memory"]="python3 src/tests/test_memory.py"
)

# -------------------------------
# Help and usage
# -------------------------------
show_help() {
  cat << EOF
Usage: $0 [OPTIONS]

Weight Perturbation Library - Modular Testing Framework

OPTIONS:
  -h, --help              Show this help message
  -l, --list              List all available tests
  -v, --verbose           Enable verbose output
  -n, --dry-run           Show what would be run without executing
  -f, --fast              Run only fast tests (skip slow ones)
  --skip-deps             Skip dependency checking
  
TEST SELECTION:
  --all                   Run all tests (default)
  --unit                  Run unit tests only
  --integration          Run integration tests only
  --congestion           Run congestion theory tests only
  --comprehensive        Run comprehensive examples only
  --examples             Run basic examples only
  --examples-congestion  Run congestion visualization examples only
  --performance          Run performance tests only
  --memory               Run memory tests only
  
INDIVIDUAL TESTS:
  --test <n>          Run specific test (can be used multiple times)
                         Examples: --test models --test integration
  
EXCLUSIONS:
  --exclude <n>       Exclude specific test (can be used multiple times)
                         Examples: --exclude memory --exclude performance
  
EXAMPLES:
  $0                                    # Run all tests
  $0 --unit --integration              # Run only unit and integration tests
  $0 --test models --test section2     # Run only models and section2 tests
  $0 --all --exclude memory            # Run all tests except memory tests
  $0 --fast --verbose                  # Run fast tests with verbose output
  $0 --list                            # List all available tests

AVAILABLE TEST CATEGORIES:
EOF

  for category in "${!TEST_CATEGORIES[@]}"; do
    printf "  %-20s %s\n" "$category" "${TEST_CATEGORIES[$category]}"
  done
}

show_test_list() {
  print_status "Available Tests by Category:" $BLUE
  echo
  
  print_status "UNIT TESTS:" $CYAN
  for test in "${!UNIT_TESTS[@]}"; do
    printf "  %-20s %s\n" "$test" "${UNIT_TESTS[$test]}"
  done
  echo
  
  print_status "INTEGRATION TESTS:" $CYAN
  for test in "${!INTEGRATION_TESTS[@]}"; do
    printf "  %-20s %s\n" "$test" "${INTEGRATION_TESTS[$test]}"
  done
  echo
  
  print_status "CONGESTION TESTS:" $CYAN
  for test in "${!CONGESTION_TESTS[@]}"; do
    printf "  %-20s %s\n" "$test" "${CONGESTION_TESTS[$test]}"
  done
  echo
  
  print_status "COMPREHENSIVE TESTS:" $CYAN
  for test in "${!COMPREHENSIVE_TESTS[@]}"; do
    printf "  %-20s %s\n" "$test" "Special comprehensive test"
  done
  echo
  
  print_status "EXAMPLE TESTS:" $CYAN
  for test in "${!EXAMPLE_TESTS[@]}"; do
    printf "  %-20s %s\n" "$test" "${EXAMPLE_TESTS[$test]}"
  done
  echo
  
  print_status "EXAMPLE CONGESTION TESTS:" $CYAN
  for test in "${!EXAMPLE_CONGESTION_TESTS[@]}"; do
    printf "  %-20s %s\n" "$test" "${EXAMPLE_CONGESTION_TESTS[$test]}"
  done
  echo
  
  print_status "PERFORMANCE TESTS:" $CYAN
  for test in "${!PERFORMANCE_TESTS[@]}"; do
    printf "  %-20s %s\n" "$test" "${PERFORMANCE_TESTS[$test]}"
  done
  echo
  
  print_status "MEMORY TESTS:" $CYAN
  for test in "${!MEMORY_TESTS[@]}"; do
    printf "  %-20s %s\n" "$test" "${MEMORY_TESTS[$test]}"
  done
}

# -------------------------------
# Argument parsing
# -------------------------------
parse_arguments() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      -h|--help)
        show_help
        exit 0
        ;;
      -l|--list)
        show_test_list
        exit 0
        ;;
      -v|--verbose)
        VERBOSE=true
        shift
        ;;
      -n|--dry-run)
        DRY_RUN=true
        shift
        ;;
      -f|--fast)
        FAST_MODE=true
        shift
        ;;
      --skip-deps)
        SKIP_DEPS=true
        shift
        ;;
      --all)
        RUN_ALL=true
        shift
        ;;
      --unit)
        RUN_UNIT=true
        RUN_ALL=false
        shift
        ;;
      --integration)
        RUN_INTEGRATION=true
        RUN_ALL=false
        shift
        ;;
      --congestion)
        RUN_CONGESTION=true
        RUN_ALL=false
        shift
        ;;
      --comprehensive)
        RUN_COMPREHENSIVE=true
        RUN_ALL=false
        shift
        ;;
      --examples)
        RUN_EXAMPLES=true
        RUN_ALL=false
        shift
        ;;
      --examples-congestion)
        RUN_EXAMPLES_CONGESTION=true
        RUN_ALL=false
        shift
        ;;
      --performance)
        RUN_PERFORMANCE=true
        RUN_ALL=false
        shift
        ;;
      --memory)
        RUN_MEMORY=true
        RUN_ALL=false
        shift
        ;;
      --test)
        if [[ -n "${2:-}" ]]; then
          INDIVIDUAL_TESTS+=("$2")
          RUN_ALL=false
          shift 2
        else
          print_status "Error: --test requires a test name" $RED
          exit 1
        fi
        ;;
      --exclude)
        if [[ -n "${2:-}" ]]; then
          EXCLUDE_TESTS+=("$2")
          shift 2
        else
          print_status "Error: --exclude requires a test name" $RED
          exit 1
        fi
        ;;
      *)
        print_status "Error: Unknown option $1" $RED
        echo "Use --help for usage information"
        exit 1
        ;;
    esac
  done
}

# -------------------------------
# Helper functions
# -------------------------------
is_excluded() {
  local test_name="$1"
  for excluded in "${EXCLUDE_TESTS[@]}"; do
    if [[ "$test_name" == "$excluded" ]]; then
      return 0
    fi
  done
  return 1
}

is_fast_test() {
  local test_name="$1"
  # Define which tests are considered "fast"
  case "$test_name" in
    models|samplers|losses|perturbation|pretrain|integration|congestion)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

should_run_test() {
  local test_name="$1"
  
  # Check if excluded
  if is_excluded "$test_name"; then
    print_debug "Test $test_name is excluded"
    return 1
  fi
  
  # Check fast mode
  if [[ "$FAST_MODE" == "true" ]] && ! is_fast_test "$test_name"; then
    print_debug "Test $test_name is not a fast test, skipping in fast mode"
    return 1
  fi
  
  return 0
}

# -------------------------------
# Dependency Check
# -------------------------------
check_dependencies() {
  if [[ "$SKIP_DEPS" == "true" ]]; then
    print_status "Skipping dependency check..." $YELLOW
    return 0
  fi

  print_status "Checking Python dependencies..." $NC

  check_dep() {
    local pkg="$1"
    print_debug "Checking dependency: $pkg"
    
    if python3 -c "import ${pkg}" >/dev/null 2>&1; then
      print_status "✓ ${pkg}" $GREEN
      return 0
    else
      print_status "✗ Missing dependency: ${pkg}" $RED
      return 1
    fi
  }

  local deps_failed=false
  
  print_debug "Checking core dependencies..."
  for dep in torch numpy matplotlib seaborn scipy yaml pathlib; do
    if ! check_dep "$dep"; then
      deps_failed=true
    fi
  done

  if [[ "$deps_failed" == "true" ]]; then
    print_status "Core dependencies missing. Please install them first." $RED
    exit 1
  fi

  # Optional deps
  print_debug "Checking optional dependencies..."
  for dep in geomloss pytest; do
    if python3 -c "import ${dep}" >/dev/null 2>&1; then
      print_status "✓ ${dep}" $GREEN
    else
      print_status "⚠ ${dep} not found (some tests may be skipped)" $YELLOW
    fi
  done

  print_status "Dependencies check completed!" $GREEN
}

# -------------------------------
# Test execution helpers
# -------------------------------
run_test() {
  local name="$1"
  local cmd="$2"
  local log_file="${LOG_DIR}/${name}.log"
  local timeout_seconds="${3:-120}"  # Default 2 minutes timeout

  print_debug "Checking if test $name should run..."
  
  if ! should_run_test "$name"; then
    print_debug "Skipping ${name} (excluded or not in fast mode)"
    return 0
  fi

  TOTAL_TESTS+=1
  
  if [[ "$DRY_RUN" == "true" ]]; then
    print_status "[DRY RUN] Would run ${name}: ${cmd}" $CYAN
    return 0
  fi

  print_status "Running ${name} (timeout: ${timeout_seconds}s)..." $YELLOW
  print_debug "Command: ${cmd}"
  print_debug "Log file: ${log_file}"
  
  local start_time=$(date +%s)
  local exit_code=0
  
  # Run with timeout
  if command -v timeout >/dev/null 2>&1; then
    timeout ${timeout_seconds}s bash -c "${cmd}" > "${log_file}" 2>&1
    exit_code=$?
    if [[ $exit_code -eq 124 ]]; then
      local end_time=$(date +%s)
      local duration=$((end_time - start_time))
      print_status "✗ ${name} FAILED (timeout after ${timeout_seconds}s)" $RED
      FAILED+=1
      TEST_RESULTS+=("✗ ${name} - FAILED (timeout after ${timeout_seconds}s)")
      echo "Check log file: ${log_file}"
      return 1
    fi
  else
    # Fallback for systems without timeout command
    eval "${cmd}" > "${log_file}" 2>&1 &
    local bg_pid=$!
    local count=0
    while kill -0 $bg_pid 2>/dev/null && [[ $count -lt $timeout_seconds ]]; do
      sleep 1
      ((count++))
      if [[ $((count % 30)) -eq 0 ]]; then
        print_debug "${name} still running... (${count}s elapsed)"
      fi
    done
    
    if kill -0 $bg_pid 2>/dev/null; then
      print_status "✗ ${name} FAILED (timeout after ${timeout_seconds}s)" $RED
      kill -TERM $bg_pid 2>/dev/null || kill -KILL $bg_pid 2>/dev/null
      wait $bg_pid 2>/dev/null || true
      FAILED+=1
      TEST_RESULTS+=("✗ ${name} - FAILED (timeout)")
      echo "Check log file: ${log_file}"
      return 1
    fi
    
    wait $bg_pid
    exit_code=$?
  fi
  
  local end_time=$(date +%s)
  local duration=$((end_time - start_time))
  
  if [[ $exit_code -eq 0 ]]; then
    print_status "✓ ${name} PASSED (${duration}s)" $GREEN
    PASSED+=1
    TEST_RESULTS+=("✓ ${name} - PASSED (${duration}s)")
  else
    print_status "✗ ${name} FAILED (${duration}s)" $RED
    FAILED+=1
    TEST_RESULTS+=("✗ ${name} - FAILED (${duration}s)")
    if [[ "$VERBOSE" == "true" ]]; then
      echo "Last 10 lines of log:"
      tail -n 10 "${log_file}" || true
    else
      echo "Check log file: ${log_file}"
    fi
  fi
}

# -------------------------------
# Special comprehensive test
# -------------------------------
test_comprehensive() {
  local name="comprehensive"
  
  print_debug "Checking if comprehensive test should run..."
  
  if ! should_run_test "$name"; then
    print_debug "Skipping ${name} (excluded or not in fast mode)"
    return 0
  fi

  # Note: TOTAL_TESTS is incremented by the caller (run_comprehensive_tests)
  
  if [[ "$DRY_RUN" == "true" ]]; then
    print_status "[DRY RUN] Would run comprehensive test with special handling" $CYAN
    return 0
  fi

  # Check if the comprehensive test file exists
  local test_file="src/tests/test_comprehensive_examples.py"
  if [[ ! -f "$test_file" ]]; then
    print_status "✗ comprehensive FAILED (test file not found: $test_file)" $RED
    FAILED+=1
    TEST_RESULTS+=("✗ comprehensive - FAILED (test file not found)")
    return 1
  fi

  print_status "Running comprehensive (with 300s timeout)..." $YELLOW

  local log_file="${LOG_DIR}/comprehensive.log"
  local start_time=$(date +%s)
  
  print_debug "Running comprehensive test with timeout..."
  
  # Run the test with real-time output and timeout (300 seconds = 5 minutes)
  local exit_code=0
  
  if command -v timeout >/dev/null 2>&1; then
    print_debug "Using timeout command for comprehensive test"
    if [[ "$VERBOSE" == "true" ]]; then
      # Real-time output with tee (simplified)
      timeout 300s python3 src/tests/test_comprehensive_examples.py --verbose | tee "${log_file}"
      exit_code=${PIPESTATUS[0]}
    else
      timeout 300s python3 src/tests/test_comprehensive_examples.py --verbose > "${log_file}" 2>&1
      exit_code=$?
    fi
    
    if [[ $exit_code -eq 124 ]]; then
      print_status "✗ comprehensive FAILED (timeout after 300s)" $RED
      FAILED+=1
      TEST_RESULTS+=("✗ comprehensive - FAILED (timeout)")
      return 1
    fi
  else
    print_debug "Using fallback timeout for comprehensive test"
    # Fallback for systems without timeout command
    if [[ "$VERBOSE" == "true" ]]; then
      python3 src/tests/test_comprehensive_examples.py --verbose | tee "${log_file}" &
    else
      python3 src/tests/test_comprehensive_examples.py --verbose > "${log_file}" 2>&1 &
    fi
    
    local bg_pid=$!
    local count=0
    while kill -0 $bg_pid 2>/dev/null && [[ $count -lt 300 ]]; do
      sleep 1
      ((count++))
      if [[ $((count % 30)) -eq 0 ]]; then
        print_status "Comprehensive test still running... (${count}s elapsed)" $CYAN
      fi
    done
    
    if kill -0 $bg_pid 2>/dev/null; then
      print_status "✗ comprehensive FAILED (timeout after 300s)" $RED
      kill -TERM $bg_pid 2>/dev/null || kill -KILL $bg_pid 2>/dev/null
      wait $bg_pid 2>/dev/null || true
      FAILED+=1
      TEST_RESULTS+=("✗ comprehensive - FAILED (timeout)")
      return 1
    fi
    
    wait $bg_pid
    exit_code=$?
  fi

  local end_time=$(date +%s)
  local duration=$((end_time - start_time))
  
  local report="${LOG_DIR}/comprehensive_test_report.md"
  local failed_error_handling=0
  local total_failed=0
  local failed_others=0

  print_debug "Exit code: $exit_code, Duration: ${duration}s"
  print_debug "Checking report file: ${report}"
  print_debug "Checking log file: ${log_file}"

  # Parse report or log
  if [[ -f "${report}" ]]; then
    print_debug "Found report file: ${report}"
    if grep -qE '^- \*\*Error Handling:\*\* .*FAILED' "${report}"; then
      failed_error_handling=1
      print_debug "Found Error Handling failure in report"
    fi
    total_failed=$(grep -Ec 'FAILED' "${report}" || true)
    failed_others=$(( total_failed - failed_error_handling ))
    print_debug "Report: total_failed=$total_failed, failed_error_handling=$failed_error_handling, failed_others=$failed_others"
  else
    print_debug "No report file found, parsing log file: ${log_file}"
    # Check for Error Handling failure in log
    if grep -qE 'Error Handling.*FAILED' "${log_file}"; then
      failed_error_handling=1
      print_debug "Found Error Handling failure in log"
    fi
    # Count total failures in the summary section
    total_failed=$(grep -Ec 'FAILED' "${log_file}" || true)
    
    # Alternative: check the summary line for actual count
    if grep -qE 'Total: [0-9]+/[0-9]+ passed' "${log_file}"; then
      local summary_line=$(grep -E 'Total: [0-9]+/[0-9]+ passed' "${log_file}")
      print_debug "Found summary line: $summary_line"
      # Extract the numbers (e.g., "Total: 5/6 passed" -> passed=5, total=6, failed=1)
      local passed_count=$(echo "$summary_line" | sed -E 's/.*Total: ([0-9]+)\/([0-9]+) passed.*/\1/')
      local total_count=$(echo "$summary_line" | sed -E 's/.*Total: ([0-9]+)\/([0-9]+) passed.*/\2/')
      total_failed=$((total_count - passed_count))
      print_debug "Extracted from summary: passed=$passed_count, total=$total_count, failed=$total_failed"
    fi
    
    failed_others=$(( total_failed - failed_error_handling ))
    print_debug "Log: total_failed=$total_failed, failed_error_handling=$failed_error_handling, failed_others=$failed_others"
  fi

  # Special policy for comprehensive test:
  # - It should have exactly 1 failure (Error Handling)
  # - Exit code should be non-zero due to that failure
  # - All other tests should pass

  print_debug "=== COMPREHENSIVE TEST DECISION LOGIC ==="
  print_debug "Exit code: $exit_code"
  print_debug "Duration: ${duration}s"
  print_debug "Total failed: $total_failed"
  print_debug "Failed error handling: $failed_error_handling"
  print_debug "Failed others: $failed_others"

  # Check if it had 0 fails (unexpected success)
  if [[ $exit_code -eq 0 ]]; then
    print_status "✗ comprehensive FAILED (expected exactly 1 fail, found 0) (${duration}s)" $RED
    FAILED+=1
    TEST_RESULTS+=("✗ comprehensive - FAILED (expected exactly 1 fail, found 0) (${duration}s)")
    print_debug "DECISION: FAILED - exit code was 0 (no failures detected)"
    if [[ "$VERBOSE" == "true" ]]; then
      echo "Last 10 lines of log:"
      tail -n 10 "${log_file}" || true
    else
      echo "Check log file: ${log_file}"
    fi
    return 1
  fi

  # Check policy: pass only if exactly Error Handling failed
  # Accept if we have exactly 1 total failure and Error Handling was detected as failed
  if [[ $total_failed -eq 1 && $failed_error_handling -eq 1 ]]; then
    print_status "✓ comprehensive PASSED WITH 1 EXPECTED FAIL (${duration}s)" $GREEN
    PASSED+=1
    TEST_RESULTS+=("✓ comprehensive - PASSED WITH 1 EXPECTED FAIL (${duration}s)")
    print_debug "DECISION: PASSED - exactly 1 failure in Error Handling"
    return 0
  # Alternative check: if we found Error Handling failure and no other failures
  elif [[ $failed_error_handling -eq 1 && $failed_others -eq 0 ]]; then
    print_status "✓ comprehensive PASSED WITH 1 EXPECTED FAIL (${duration}s)" $GREEN
    PASSED+=1
    TEST_RESULTS+=("✓ comprehensive - PASSED WITH 1 EXPECTED FAIL (${duration}s)")
    print_debug "DECISION: PASSED - Error Handling failed, no other failures"
    return 0
  else
    local message="✗ comprehensive FAILED (${duration}s)"
    if [[ $failed_error_handling -ne 1 ]]; then
      message="$message (Error Handling not failed as expected)"
    fi
    if [[ $failed_others -gt 0 ]]; then
      if [[ $failed_others -eq 1 ]]; then
        message="$message (found 1 additional fail)"
      else
        message="$message (found $failed_others additional fails)"
      fi
    fi
    if [[ $total_failed -ne 1 ]]; then
      message="$message (expected exactly 1 failure, found $total_failed)"
    fi
    print_status "$message" $RED
    FAILED+=1
    TEST_RESULTS+=("$message")
    print_debug "DECISION: FAILED - conditions not met"
    if [[ "$VERBOSE" == "true" ]]; then
      echo "Last 10 lines of log:"
      tail -n 10 "${log_file}" || true
    else
      echo "Check log file: ${log_file}"
    fi
    return 1
  fi
}

# -------------------------------
# Individual test selection helpers
# -------------------------------
should_run_individual_test() {
  local test_name="$1"
  
  # If individual tests are specified, check if this test is in the list
  if [[ ${#INDIVIDUAL_TESTS[@]} -gt 0 ]]; then
    for individual_test in "${INDIVIDUAL_TESTS[@]}"; do
      if [[ "$test_name" == "$individual_test" ]]; then
        print_debug "Individual test $test_name found in INDIVIDUAL_TESTS"
        return 0
      fi
    done
    print_debug "Individual test $test_name not found in INDIVIDUAL_TESTS"
    return 1
  fi
  
  # If no individual tests specified, return true 
  # (category-level decisions are made in the run_*_tests functions)
  print_debug "No individual tests specified, allowing $test_name"
  return 0
}

# -------------------------------
# Test category runners
# -------------------------------
run_unit_tests() {
  print_debug "Checking if unit tests should run..."
  
  local should_run=false
  
  if [[ "$RUN_UNIT" == "true" ]]; then
    print_debug "Unit tests selected via --unit flag"
    should_run=true
  elif [[ "$RUN_ALL" == "true" ]]; then
    print_debug "Unit tests selected via --all flag"  
    should_run=true
  elif [[ ${#INDIVIDUAL_TESTS[@]} -gt 0 ]]; then
    # Check if any individual test belongs to unit category
    for test in "${INDIVIDUAL_TESTS[@]}"; do
      if [[ -n "${UNIT_TESTS[$test]:-}" ]]; then
        print_debug "Unit tests selected via --test $test"
        should_run=true
        break
      fi
    done
  fi
  
  if [[ "$should_run" == "true" ]]; then
    print_status "=== UNIT TESTS ===" $BLUE
    for test_name in "${!UNIT_TESTS[@]}"; do
      if should_run_individual_test "$test_name"; then
        run_test "${test_name}_test" "${UNIT_TESTS[$test_name]}" 60  # 1 minute timeout
      fi
    done
  else
    print_debug "Skipping unit tests"
  fi
}

run_integration_tests() {
  print_debug "Checking if integration tests should run..."
  
  local should_run=false
  
  if [[ "$RUN_INTEGRATION" == "true" ]]; then
    print_debug "Integration tests selected via --integration flag"
    should_run=true
  elif [[ "$RUN_ALL" == "true" ]]; then
    print_debug "Integration tests selected via --all flag"
    should_run=true
  elif [[ ${#INDIVIDUAL_TESTS[@]} -gt 0 ]]; then
    # Check if any individual test belongs to integration category
    for test in "${INDIVIDUAL_TESTS[@]}"; do
      if [[ -n "${INTEGRATION_TESTS[$test]:-}" ]]; then
        print_debug "Integration tests selected via --test $test"
        should_run=true
        break
      fi
    done
  fi
  
  if [[ "$should_run" == "true" ]]; then
    print_status "=== INTEGRATION TESTS ===" $BLUE
    for test_name in "${!INTEGRATION_TESTS[@]}"; do
      if should_run_individual_test "$test_name"; then
        run_test "${test_name}_test" "${INTEGRATION_TESTS[$test_name]}" 180  # 3 minutes timeout
      fi
    done
  else
    print_debug "Skipping integration tests"
  fi
}

run_congestion_tests() {
  print_debug "Checking if congestion tests should run..."
  
  local should_run=false
  
  if [[ "$RUN_CONGESTION" == "true" ]]; then
    print_debug "Congestion tests selected via --congestion flag"
    should_run=true
  elif [[ "$RUN_ALL" == "true" ]]; then
    print_debug "Congestion tests selected via --all flag"
    should_run=true
  elif [[ ${#INDIVIDUAL_TESTS[@]} -gt 0 ]]; then
    # Check if any individual test belongs to congestion category
    for test in "${INDIVIDUAL_TESTS[@]}"; do
      if [[ -n "${CONGESTION_TESTS[$test]:-}" ]]; then
        print_debug "Congestion tests selected via --test $test"
        should_run=true
        break
      fi
    done
  fi
  
  if [[ "$should_run" == "true" ]]; then
    print_status "=== CONGESTION THEORY TESTS ===" $BLUE
    for test_name in "${!CONGESTION_TESTS[@]}"; do
      if should_run_individual_test "$test_name"; then
        run_test "${test_name}_test" "${CONGESTION_TESTS[$test_name]}" 300  # 5 minutes timeout
      fi
    done
  else
    print_debug "Skipping congestion tests"
  fi
}

run_comprehensive_tests() {
  print_debug "Checking if comprehensive tests should run..."
  
  # Only run comprehensive if:
  # 1. Explicitly requested with --comprehensive
  # 2. Running all tests (--all or default)
  # 3. Individual test "comprehensive" is specified
  local should_run=false
  
  if [[ "$RUN_COMPREHENSIVE" == "true" ]]; then
    print_debug "Comprehensive selected via --comprehensive flag"
    should_run=true
  elif [[ "$RUN_ALL" == "true" ]]; then
    print_debug "Comprehensive selected via --all flag"
    should_run=true
  elif [[ ${#INDIVIDUAL_TESTS[@]} -gt 0 ]]; then
    # Check if "comprehensive" is in individual tests
    for test in "${INDIVIDUAL_TESTS[@]}"; do
      if [[ "$test" == "comprehensive" ]]; then
        print_debug "Comprehensive selected via --test comprehensive"
        should_run=true
        break
      fi
    done
  fi
  
  if [[ "$should_run" == "true" ]]; then
    print_status "=== COMPREHENSIVE EXAMPLES ===" $BLUE
    
    # Special handling for comprehensive test (like original script)
    TOTAL_TESTS+=1
    if test_comprehensive; then
      # test_comprehensive already handled PASSED and TEST_RESULTS
      print_debug "Comprehensive test passed with expected failure"
    else
      # test_comprehensive already handled FAILED and TEST_RESULTS  
      print_debug "Comprehensive test failed unexpectedly"
    fi
  else
    print_debug "Skipping comprehensive tests"
  fi
}

run_example_tests() {
  print_debug "Checking if example tests should run..."
  
  local should_run=false
  
  if [[ "$RUN_EXAMPLES" == "true" ]]; then
    print_debug "Example tests selected via --examples flag"
    should_run=true
  elif [[ "$RUN_ALL" == "true" ]]; then
    print_debug "Example tests selected via --all flag"
    should_run=true
  elif [[ ${#INDIVIDUAL_TESTS[@]} -gt 0 ]]; then
    # Check if any individual test belongs to examples category
    for test in "${INDIVIDUAL_TESTS[@]}"; do
      if [[ -n "${EXAMPLE_TESTS[$test]:-}" ]]; then
        print_debug "Example tests selected via --test $test"
        should_run=true
        break
      fi
    done
  fi
  
  if [[ "$should_run" == "true" ]]; then
    print_status "=== BASIC EXAMPLES ===" $BLUE
    for test_name in "${!EXAMPLE_TESTS[@]}"; do
      if should_run_individual_test "$test_name"; then
        run_test "${test_name}_example" "${EXAMPLE_TESTS[$test_name]}" 120  # 2 minutes timeout
      fi
    done
  else
    print_debug "Skipping example tests"
  fi
}

run_example_congestion_tests() {
  print_debug "Checking if example congestion tests should run..."
  
  local should_run=false
  
  if [[ "$RUN_EXAMPLES_CONGESTION" == "true" ]]; then
    print_debug "Example congestion tests selected via --examples-congestion flag"
    should_run=true
  elif [[ "$RUN_ALL" == "true" ]]; then
    print_debug "Example congestion tests selected via --all flag"
    should_run=true
  elif [[ ${#INDIVIDUAL_TESTS[@]} -gt 0 ]]; then
    # Check if any individual test belongs to examples-congestion category
    for test in "${INDIVIDUAL_TESTS[@]}"; do
      if [[ -n "${EXAMPLE_CONGESTION_TESTS[$test]:-}" ]]; then
        print_debug "Example congestion tests selected via --test $test"
        should_run=true
        break
      fi
    done
  fi
  
  if [[ "$should_run" == "true" ]]; then
    print_status "=== CONGESTION EXAMPLES WITH VISUALIZATION ===" $BLUE
    for test_name in "${!EXAMPLE_CONGESTION_TESTS[@]}"; do
      if should_run_individual_test "$test_name"; then
        run_test "${test_name}" "${EXAMPLE_CONGESTION_TESTS[$test_name]}" 180  # 3 minutes timeout
      fi
    done
  else
    print_debug "Skipping example congestion tests"
  fi
}

run_performance_tests() {
  print_debug "Checking if performance tests should run..."
  
  local should_run=false
  
  if [[ "$RUN_PERFORMANCE" == "true" ]]; then
    print_debug "Performance tests selected via --performance flag"
    should_run=true
  elif [[ "$RUN_ALL" == "true" ]]; then
    print_debug "Performance tests selected via --all flag"
    should_run=true
  elif [[ ${#INDIVIDUAL_TESTS[@]} -gt 0 ]]; then
    # Check if any individual test belongs to performance category
    for test in "${INDIVIDUAL_TESTS[@]}"; do
      if [[ -n "${PERFORMANCE_TESTS[$test]:-}" ]]; then
        print_debug "Performance tests selected via --test $test"
        should_run=true
        break
      fi
    done
  fi
  
  if [[ "$should_run" == "true" ]]; then
    print_status "=== PERFORMANCE TESTS ===" $BLUE
    for test_name in "${!PERFORMANCE_TESTS[@]}"; do
      if should_run_individual_test "$test_name"; then
        run_test "${test_name}_test" "${PERFORMANCE_TESTS[$test_name]}" 600  # 10 minutes timeout
      fi
    done
  else
    print_debug "Skipping performance tests"
  fi
}

run_memory_tests() {
  print_debug "Checking if memory tests should run..."
  
  local should_run=false
  
  if [[ "$RUN_MEMORY" == "true" ]]; then
    print_debug "Memory tests selected via --memory flag"
    should_run=true
  elif [[ "$RUN_ALL" == "true" ]]; then
    print_debug "Memory tests selected via --all flag"
    should_run=true
  elif [[ ${#INDIVIDUAL_TESTS[@]} -gt 0 ]]; then
    # Check if any individual test belongs to memory category
    for test in "${INDIVIDUAL_TESTS[@]}"; do
      if [[ -n "${MEMORY_TESTS[$test]:-}" ]]; then
        print_debug "Memory tests selected via --test $test"
        should_run=true
        break
      fi
    done
  fi
  
  if [[ "$should_run" == "true" ]]; then
    print_status "=== MEMORY TESTS ===" $BLUE
    for test_name in "${!MEMORY_TESTS[@]}"; do
      if should_run_individual_test "$test_name"; then
        run_test "${test_name}_test" "${MEMORY_TESTS[$test_name]}" 300  # 5 minutes timeout
      fi
    done
  else
    print_debug "Skipping memory tests"
  fi
}

# -------------------------------
# Summary generation
# -------------------------------
generate_summary() {
  local END_TS=$(date +%s)
  local DURATION=$(( END_TS - START_TS ))

  print_status "=============================================================" $BLUE
  print_status "TEST SUITE COMPLETED" $BLUE
  print_status "=============================================================" $BLUE
  echo "Duration: ${DURATION}s"
  echo "Total tests: ${TOTAL_TESTS}"
  echo "Passed: ${PASSED}"
  echo "Failed: ${FAILED}"
  
  if [[ "$FAST_MODE" == "true" ]]; then
    echo "Mode: Fast tests only"
  fi
  
  if [[ ${#EXCLUDE_TESTS[@]} -gt 0 ]]; then
    echo "Excluded: ${EXCLUDE_TESTS[*]}"
  fi
  
  print_status "=============================================================" $BLUE

  # Save summary
  {
    echo "Weight Perturbation Library - Test Summary"
    echo "=========================================="
    echo "Date: $(date)"
    echo "Duration: ${DURATION}s"
    echo "Total tests run: ${TOTAL_TESTS}"
    echo "Passed: ${PASSED}"
    echo "Failed: ${FAILED}"
    
    if [[ "$FAST_MODE" == "true" ]]; then
      echo "Mode: Fast tests only"
    fi
    
    if [[ ${#EXCLUDE_TESTS[@]} -gt 0 ]]; then
      echo "Excluded: ${EXCLUDE_TESTS[*]}"
    fi
    
    if [[ ${#INDIVIDUAL_TESTS[@]} -gt 0 ]]; then
      echo "Individual tests: ${INDIVIDUAL_TESTS[*]}"
    fi
    
    echo
    echo "Test Results:"
    for result in "${TEST_RESULTS[@]}"; do
      echo "${result}"
    done
  } > "${SUMMARY_FILE}"

  # Save failed list for quick inspection
  {
    for result in "${TEST_RESULTS[@]}"; do
      if [[ "${result}" == \✗* ]]; then
        echo "${result}"
      fi
    done
  } > "${FAILED_LIST_FILE}"

  # Final status line
  if [[ ${FAILED} -gt 0 ]]; then
    print_status "❌ Some tests failed. Check logs in ${LOG_DIR}/" $RED
    exit 1
  else
    if [[ ${TOTAL_TESTS} -eq 0 ]]; then
      print_status "⚠️  No tests were run!" $YELLOW
      exit 1
    else
      print_status "✅ All tests passed successfully!" $GREEN
      exit 0
    fi
  fi
}

# -------------------------------
# Main execution
# -------------------------------
main() {
  print_debug "Starting main function..."
  
  # Parse command line arguments
  parse_arguments "$@"
  
  print_debug "Arguments parsed. VERBOSE=$VERBOSE, DRY_RUN=$DRY_RUN, FAST_MODE=$FAST_MODE"
  
  # Setup directories
  mkdir -p "${LOG_DIR}" "${PLOT_DIR}"
  print_debug "Created directories: ${LOG_DIR}, ${PLOT_DIR}"
  
  # Show banner
  print_status "==============================================================" $BLUE
  print_status "WEIGHT PERTURBATION LIBRARY - CONGESTED TRANSPORT TESTING" $BLUE
  print_status "==============================================================" $BLUE
  
  if [[ "$DRY_RUN" == "true" ]]; then
    print_status "DRY RUN MODE - No tests will be executed" $YELLOW
  fi
  
  if [[ "$FAST_MODE" == "true" ]]; then
    print_status "FAST MODE - Only running fast tests" $CYAN
  fi
  
  if [[ ${#EXCLUDE_TESTS[@]} -gt 0 ]]; then
    print_status "EXCLUDING: ${EXCLUDE_TESTS[*]}" $YELLOW
  fi
  
  if [[ ${#INDIVIDUAL_TESTS[@]} -gt 0 ]]; then
    print_status "RUNNING INDIVIDUAL TESTS: ${INDIVIDUAL_TESTS[*]}" $CYAN
  fi
  
  print_status "Starting testing suite..." $NC
  
  # Check dependencies
  print_debug "Checking dependencies..."
  check_dependencies
  print_debug "Dependencies checked successfully"
  
  # Run test categories
  print_debug "Starting test execution..."
  run_unit_tests
  run_integration_tests
  run_congestion_tests
  run_comprehensive_tests
  run_example_tests
  run_example_congestion_tests
  run_performance_tests
  run_memory_tests
  
  print_debug "All test categories completed"
  
  # Generate summary
  generate_summary
}

# -------------------------------
# Script entry point
# -------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi