cmake_minimum_required(VERSION 3.25)
project(qr_apple_silicon VERSION 1.0.0 LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MLX REQUIRED)

# =============================================================================
# Metal shader compilation
# =============================================================================

# QR_Unblocked  — dispatched when M < 512
set(_unblocked_metallib "${CMAKE_CURRENT_SOURCE_DIR}/QR_Unblocked.metallib")
if (NOT EXISTS "${_unblocked_metallib}")
    message(STATUS "Compiling QR_Unblocked.metal → metallib")
    execute_process(
        COMMAND xcrun -sdk macosx metal -c
                "${CMAKE_CURRENT_SOURCE_DIR}/QR_Unblocked.metal"
                -o "${CMAKE_CURRENT_BINARY_DIR}/QR_Unblocked.air"
        COMMAND_ERROR_IS_FATAL ANY
    )
    execute_process(
        COMMAND xcrun -sdk macosx metallib
                "${CMAKE_CURRENT_BINARY_DIR}/QR_Unblocked.air"
                -o "${_unblocked_metallib}"
        COMMAND_ERROR_IS_FATAL ANY
    )
endif ()

# QR_Streaming_AMX — dispatched when M >= 512
set(_streaming_amx_reduced_metallib "${CMAKE_CURRENT_SOURCE_DIR}/QR_Streaming_AMX_Reduced.metallib")
if (NOT EXISTS "${_streaming_amx_reduced_metallib}")
    message(STATUS "Compiling QR_Streaming_AMX_Reduced.metal → metallib")
    execute_process(
        COMMAND xcrun -sdk macosx metal -c
                "${CMAKE_CURRENT_SOURCE_DIR}/QR_Streaming_AMX_Reduced.metal"
                -o "${CMAKE_CURRENT_BINARY_DIR}/QR_Streaming_AMX_Reduced.air"
        COMMAND_ERROR_IS_FATAL ANY
    )
    execute_process(
        COMMAND xcrun -sdk macosx metallib
                "${CMAKE_CURRENT_BINARY_DIR}/QR_Streaming_AMX_Reduced.air"
                -o "${_streaming_amx_reduced_metallib}"
        COMMAND_ERROR_IS_FATAL ANY
    )
endif ()

set(_streaming_amx_complete_metallib "${CMAKE_CURRENT_SOURCE_DIR}/QR_Streaming_AMX_Complete.metallib")
if (NOT EXISTS "${_streaming_amx_complete_metallib}")
    message(STATUS "Compiling QR_Streaming_AMX_Complete.metal → metallib")
    execute_process(
        COMMAND xcrun -sdk macosx metal -c
                "${CMAKE_CURRENT_SOURCE_DIR}/QR_Streaming_AMX_Complete.metal"
                -o "${CMAKE_CURRENT_BINARY_DIR}/QR_Streaming_AMX_Complete.air"
        COMMAND_ERROR_IS_FATAL ANY
    )
    execute_process(
        COMMAND xcrun -sdk macosx metallib
                "${CMAKE_CURRENT_BINARY_DIR}/QR_Streaming_AMX_Complete.air"
                -o "${_streaming_amx_complete_metallib}"
        COMMAND_ERROR_IS_FATAL ANY
    )
endif ()

# =============================================================================
# qr_metal library
# =============================================================================

add_library(qr_metal STATIC
    qr_dispatch.mm
    qr_unblocked.mm
        qr_streaming_amx_reduced.mm
        qr_streaming_amx_complete.mm
)

target_include_directories(qr_metal PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(qr_metal PRIVATE
    "QR_UNBLOCKED_METALLIB_PATH=\"${_unblocked_metallib}\""
    "QR_STREAMING_AMX_REDUCED_METALLIB_PATH=\"${_streaming_amx_reduced_metallib}\""
    "QR_STREAMING_AMX_COMPLETE_METALLIB_PATH=\"${_streaming_amx_complete_metallib}\""
)
target_link_libraries(qr_metal PUBLIC
    mlx
    "-framework Metal"
    "-framework Foundation"
)

# =============================================================================
# Benchmark executable
# =============================================================================

add_executable(benchmark_qr tests/benchmark_qr.cpp)
target_link_libraries(benchmark_qr PRIVATE qr_metal mlx)
