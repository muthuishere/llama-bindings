#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_FILE="$SCRIPT_DIR/e2e-report.json"

# Flags
SKIP_GO=false; SKIP_JAVA=false; SKIP_JS=false
for arg in "$@"; do
  case "$arg" in
    --skip-go)   SKIP_GO=true ;;
    --skip-java) SKIP_JAVA=true ;;
    --skip-js)   SKIP_JS=true ;;
    *) echo "Unknown flag: $arg" >&2; exit 1 ;;
  esac
done

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; BOLD='\033[1m'; RESET='\033[0m'

now_ms() { python3 -c "import time; print(int(time.time()*1000))"; }
TIMESTAMP=$(python3 -c "from datetime import datetime,timezone; print(datetime.now(timezone.utc).isoformat())")
RUN_ID=$(python3 -c "import random; print(random.randint(10000,99999))")

RESULTS="[]"
TOTAL=0; PASSED=0; FAILED=0

add_result() {
  local target="$1" test="$2" status="$3" duration="$4" response="$5"
  # Escape response for JSON
  local escaped
  escaped=$(python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))" <<< "$response")
  RESULTS=$(python3 -c "
import json,sys
r=json.loads(sys.stdin.read())
r.append({'target':'$target','test':'$test','status':'$status','duration_ms':$duration,'response':$escaped})
print(json.dumps(r))
" <<< "$RESULTS")
  TOTAL=$((TOTAL+1))
  if [ "$status" = "pass" ]; then PASSED=$((PASSED+1)); else FAILED=$((FAILED+1)); fi
}

# Parse NDJSON: extract reply from last line or concatenate tokens
parse_ndjson_reply() {
  python3 -c "
import sys, json
tokens = []
reply = ''
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try:
        obj = json.loads(line)
        if obj.get('done'):
            reply = obj.get('reply', '')
        elif 'token' in obj:
            tokens.append(obj['token'])
    except json.JSONDecodeError:
        pass
if reply:
    print(reply)
else:
    print(''.join(tokens))
"
}

# Run a curl chat test
run_chat_test() {
  local target="$1" test_name="$2" port="$3" session="$4" message="$5" pattern="$6"
  local start_ms end_ms duration raw_response reply

  start_ms=$(now_ms)
  raw_response=$(curl -s --max-time 300 -X POST "http://localhost:${port}/chat" \
    -H "Content-Type: application/json" \
    -d "{\"session\":\"${session}\",\"message\":\"${message}\"}" 2>&1) || true
  end_ms=$(now_ms)
  duration=$((end_ms - start_ms))

  reply=$(echo "$raw_response" | parse_ndjson_reply)

  if echo "$reply" | grep -qiE "$pattern"; then
    add_result "$target" "$test_name" "pass" "$duration" "$reply"
  else
    add_result "$target" "$test_name" "fail" "$duration" "$reply"
  fi
}

# --- Go tests ---
if [ "$SKIP_GO" = false ]; then
  run_chat_test "go" "chat" 8080 "e2e-chat-go-${RUN_ID}" \
    "What is llama-bindings?" "cross-language|library|llama\.cpp"
  run_chat_test "go" "chat_tool" 8080 "e2e-tool-go-${RUN_ID}" \
    "What is the square root of 144?" "12"
fi

# --- Java tests ---
if [ "$SKIP_JAVA" = false ]; then
  run_chat_test "java" "chat" 8081 "e2e-chat-java-${RUN_ID}" \
    "What is llama-bindings?" "cross-language|library|llama\.cpp"
  run_chat_test "java" "chat_tool" 8081 "e2e-tool-java-${RUN_ID}" \
    "What is the square root of 144?" "12"
fi

# --- JS browser (Playwright) ---
if [ "$SKIP_JS" = false ]; then
  start_ms=$(now_ms)
  pw_exit=0
  pw_output=$(cd "$SCRIPT_DIR/js-browser" && npx playwright test --project=chromium specs/agent.spec.js 2>&1) || pw_exit=$?
  end_ms=$(now_ms)
  duration=$((end_ms - start_ms))

  if [ "$pw_exit" -eq 0 ]; then
    add_result "js-browser" "playwright" "pass" "$duration" "Playwright tests passed"
  else
    # Grab last few lines for context
    snippet=$(echo "$pw_output" | tail -5 | tr '\n' ' ')
    add_result "js-browser" "playwright" "fail" "$duration" "$snippet"
  fi
fi

# --- Build JSON report ---
REPORT=$(python3 -c "
import json, sys
results = json.loads(sys.stdin.read())
report = {
    'timestamp': '$TIMESTAMP',
    'results': results,
    'summary': {'total': $TOTAL, 'passed': $PASSED, 'failed': $FAILED}
}
print(json.dumps(report, indent=2))
" <<< "$RESULTS")

# --- Human-readable summary to stderr ---
{
  echo ""
  printf "${BOLD}%-12s %-14s %-8s %10s${RESET}\n" "TARGET" "TEST" "STATUS" "DURATION"
  printf "%-12s %-14s %-8s %10s\n" "------" "------" "------" "--------"
  python3 -c "
import json, sys
for r in json.loads(sys.stdin.read()):
    s = r['status']
    color = '\033[0;32m' if s == 'pass' else '\033[0;31m'
    reset = '\033[0m'
    print(f\"{r['target']:<12} {r['test']:<14} {color}{s:<8}{reset} {r['duration_ms']:>8}ms\")
" <<< "$RESULTS"
  echo ""
  if [ "$FAILED" -eq 0 ]; then
    printf "${GREEN}All $TOTAL tests passed.${RESET}\n"
  else
    printf "${RED}$FAILED of $TOTAL tests failed.${RESET}\n"
  fi
  echo ""
} >&2

# --- Output JSON ---
echo "$REPORT" | tee "$REPORT_FILE"

# --- Exit code ---
[ "$FAILED" -eq 0 ] && exit 0 || exit 1
