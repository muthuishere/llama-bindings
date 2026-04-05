# CMakeLists.txt — native bridge build
#
# Upstream llama.cpp: https://github.com/ggml-org/llama.cpp
# Use `task build-bridge` (Taskfile) rather than invoking CMake directly.

cmake_minimum_required(VERSION 3.18)
project(llama_bridge C)

set(CMAKE_C_STANDARD 11)

# Path to the cloned llama.cpp repository (https://github.com/ggml-org/llama.cpp)
set(LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/../third_party/llama.cpp-src"
    CACHE PATH "Path to the llama.cpp source tree")

# -----------------------------------------------------------------------
# Bridge sources
# -----------------------------------------------------------------------
add_library(llama_bridge SHARED
    src/llama_bridge.c
    src/llama_bridge_chat.c
    src/llama_bridge_embed.c
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
# Link llama.cpp once it is built (third_party/llama.cpp-src/build)
#
# Use whole-archive / force_load so that every symbol from the static
# libs is pulled into the shared library, making it fully self-contained.
# Users can load libllama_bridge.so / .dylib / .dll without installing
# llama.cpp separately.
# -----------------------------------------------------------------------
if(EXISTS "${LLAMA_CPP_DIR}/build/libllama.a")
    set(LLAMA_STATIC_LIBS
        "${LLAMA_CPP_DIR}/build/libllama.a"
        "${LLAMA_CPP_DIR}/build/ggml/src/libggml.a"
    )
elseif(EXISTS "${LLAMA_CPP_DIR}/build/src/libllama.a")
    set(LLAMA_STATIC_LIBS
        "${LLAMA_CPP_DIR}/build/src/libllama.a"
    )
else()
    set(LLAMA_STATIC_LIBS "")
endif()

if(LLAMA_STATIC_LIBS)
    if(APPLE)
        # macOS: -force_load includes every object file from the archive.
        foreach(_lib ${LLAMA_STATIC_LIBS})
            target_link_libraries(llama_bridge PRIVATE "-Wl,-force_load,${_lib}")
        endforeach()
    elseif(UNIX)
        # Linux / other ELF: --whole-archive / --no-whole-archive wraps the
        # static libs so every object is included in the shared library.
        target_link_libraries(llama_bridge PRIVATE
            "-Wl,--whole-archive"
            ${LLAMA_STATIC_LIBS}
            "-Wl,--no-whole-archive"
        )
    elseif(WIN32)
        # MSVC / MinGW: /WHOLEARCHIVE includes all objects.
        foreach(_lib ${LLAMA_STATIC_LIBS})
            target_link_options(llama_bridge PRIVATE "/WHOLEARCHIVE:${_lib}")
        endforeach()
        target_link_libraries(llama_bridge PRIVATE ${LLAMA_STATIC_LIBS})
    endif()
endif()

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
