#!/usr/bin/env bash
# verify-artifacts.sh — verify that all expected build artifacts exist.
#
# Usage:
#   ./scripts/verify-artifacts.sh [build-dir]
#
# build-dir defaults to "<repo-root>/build".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${1:-${REPO_ROOT}/build}"

fail() { echo "FAIL: $*" >&2; exit 1; }
ok()   { echo "OK:   $*"; }

echo "Verifying build artifacts in: ${BUILD_DIR}"
echo ""

# ---- llama.cpp shared library ----
LLAMA_LIB=""
for candidate in \
    "${BUILD_DIR}/libllama.so" \
    "${BUILD_DIR}/libllama.dylib" \
    "${BUILD_DIR}/libllama.dll" \
    "${BUILD_DIR}/src/libllama.so" \
    "${BUILD_DIR}/src/libllama.dylib";
do
    if [ -f "${candidate}" ]; then
        LLAMA_LIB="${candidate}"
        break
    fi
done

if [ -z "${LLAMA_LIB}" ]; then
    fail "libllama not found in ${BUILD_DIR}"
else
    ok "libllama found: ${LLAMA_LIB}"
fi

# ---- bridge shared library ----
BRIDGE_LIB=""
for candidate in \
    "${BUILD_DIR}/libllama_bridge.so" \
    "${BUILD_DIR}/libllama_bridge.dylib" \
    "${BUILD_DIR}/libllama_bridge.dll";
do
    if [ -f "${candidate}" ]; then
        BRIDGE_LIB="${candidate}"
        break
    fi
done

if [ -z "${BRIDGE_LIB}" ]; then
    fail "libllama_bridge not found in ${BUILD_DIR}"
else
    ok "libllama_bridge found: ${BRIDGE_LIB}"
fi

# ---- bridge header ----
HEADER="${REPO_ROOT}/bridge/include/llama_bridge.h"
if [ ! -f "${HEADER}" ]; then
    fail "bridge header not found: ${HEADER}"
else
    ok "bridge header found: ${HEADER}"
fi

# ---- Go source ----
GO_SRC="${REPO_ROOT}/bindings/go/llama/engine.go"
if [ ! -f "${GO_SRC}" ]; then
    fail "Go binding not found: ${GO_SRC}"
else
    ok "Go binding found: ${GO_SRC}"
fi

# ---- Java source ----
JAVA_SRC="${REPO_ROOT}/bindings/java/src/main/java/com/example/llama/LlamaEngine.java"
if [ ! -f "${JAVA_SRC}" ]; then
    fail "Java binding not found: ${JAVA_SRC}"
else
    ok "Java binding found: ${JAVA_SRC}"
fi

# ---- JS source ----
JS_SRC="${REPO_ROOT}/bindings/js/src/llama.js"
if [ ! -f "${JS_SRC}" ]; then
    fail "JS binding not found: ${JS_SRC}"
else
    ok "JS binding found: ${JS_SRC}"
fi

echo ""
echo "All artifact checks passed."
