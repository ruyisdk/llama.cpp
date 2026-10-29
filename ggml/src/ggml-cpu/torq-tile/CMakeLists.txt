option(GGML_CPU_TORQ_TILE_ZTT "ggml: use torq-tile Ztt Q4_K and Q6_K kernels" OFF)

set(TORQ_TILE_ZTT_COMPILE_OPTIONS "-march=rv64gcv_ztt_zttuds32b_zttm16_ztta4;-mabi=lp64d" CACHE STRING "Compiler options for torq-tile Ztt kernels")
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
string(REPLACE ";" " " CMAKE_REQUIRED_FLAGS "${TORQ_TILE_ZTT_COMPILE_OPTIONS}")
unset(TORQ_TILE_COMPILER_HAS_ZTT)
unset(TORQ_TILE_COMPILER_HAS_ZTT CACHE)
check_cxx_source_compiles("
    #include <riscv_ztt.h>
    #if !defined(__riscv_ztt) || !defined(__riscv_flen) || __riscv_flen < 32
    #error Ztt and scalar F support are required
    #endif
    #if (!defined(__riscv_zttm16) && !defined(__riscv_zttm32)) || (!defined(__riscv_ztta4) && !defined(__riscv_ztta8) && !defined(__riscv_ztta16))
    #error Unsupported Ztt register profile
    #endif
    int main() {
        __asm__ volatile(\"mmulbtacc.2d acc0, m0, m2\");
        return (int) __riscv_ztt_get_ameown();
    }
" TORQ_TILE_COMPILER_HAS_ZTT)
cmake_pop_check_state()
if (NOT TORQ_TILE_COMPILER_HAS_ZTT)
    message(FATAL_ERROR "Ztt compiler probe failed; check TORQ_TILE_ZTT_COMPILE_OPTIONS and riscv_ztt.h")
endif()

add_library(${TORQ_TILE_ZTT_KERNEL_TARGET} OBJECT
    kernels.cpp
    ${TORQ_TILE_SRC}/src/gemm/gemm_i8_u4/gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt.cpp
    ${TORQ_TILE_SRC}/src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt.cpp
)
target_compile_features(${TORQ_TILE_ZTT_KERNEL_TARGET} PRIVATE cxx_std_17)
target_compile_options(${TORQ_TILE_ZTT_KERNEL_TARGET} PRIVATE ${TORQ_TILE_ZTT_COMPILE_OPTIONS})
target_include_directories(${TORQ_TILE_ZTT_KERNEL_TARGET} PRIVATE
    ../..
    ..
    ../../../include
    ${TORQ_TILE_SRC}
)
set_target_properties(${TORQ_TILE_ZTT_KERNEL_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
