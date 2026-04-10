# CMakeLists.txt — native bridge build
#
# Upstream llama.cpp: https://github.com/ggml-org/llama.cpp
# Use `task build-bridge` (Taskfile) rather than invoking CMake directly.

cmake_minimum_required(VERSION 3.18)
project(llama_bridge CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# Path to the cloned llama.cpp repository (https://github.com/ggml-org/llama.cpp)
set(LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/../third_party/llama.cpp-src"
    CACHE PATH "Path to the llama.cpp source tree")

# -----------------------------------------------------------------------
# Bridge sources
# -----------------------------------------------------------------------
add_library(llama_bridge SHARED
    src/llama_bridge.cpp
    src/llama_bridge_chat.cpp
    src/llama_bridge_embed.cpp
    src/llama_bridge_observe.c
    src/llama_bridge_json.c
)

target_include_directories(llama_bridge
    PUBLIC  include
    PRIVATE src
            ${LLAMA_CPP_DIR}/include
            ${LLAMA_CPP_DIR}/ggml/include
)

# -----------------------------------------------------------------------
# Link against llama.cpp shared libraries (CPU-only build)
# NOTE: .dylib linking below is macOS-specific
# -----------------------------------------------------------------------
set(LLAMA_DYLIB_DIR "${LLAMA_CPP_DIR}/build/bin"
    CACHE PATH "Directory containing llama.cpp shared libraries")

target_link_libraries(llama_bridge PRIVATE
    "${LLAMA_DYLIB_DIR}/libllama.dylib"
    "${LLAMA_DYLIB_DIR}/libggml.dylib"
    "${LLAMA_DYLIB_DIR}/libggml-base.dylib"
    "${LLAMA_DYLIB_DIR}/libggml-cpu.dylib"
    "${LLAMA_DYLIB_DIR}/libggml-blas.dylib"
)

# Set rpath so bridge finds ggml dylibs in same directory at runtime
set_target_properties(llama_bridge PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# -----------------------------------------------------------------------
# WASM target (set -DTARGET_WASM=ON when building with emcmake)
# -----------------------------------------------------------------------
if(TARGET_WASM)
    target_compile_options(llama_bridge PRIVATE -O2)
    set_target_properties(llama_bridge PROPERTIES
        SUFFIX ".js"
        LINK_FLAGS "-s EXPORTED_FUNCTIONS='[_llama_chat_create,_llama_chat_infer_json,_llama_chat_destroy,_llama_embed_create,_llama_embed_infer,_llama_embed_destroy,_llama_bridge_string_free,_llama_bridge_float_free]' -s MODULARIZE=1 -s EXPORT_NAME=LlamaBridge"
    )
endif()

install(TARGETS llama_bridge
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(FILES include/llama_bridge.h DESTINATION include)
