// SPDX-FileCopyrightText: Copyright 2026 C-SKY Microsystems Co., Ltd.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>

struct ggml_compute_params;

// Shared with the generic CPU adapter; keep this header ISA-neutral.
namespace ggml::cpu::torq_tile {

struct kernel {
    bool (*supports)(const ggml_tensor * op);
    size_t (*get_work_size)(const ggml_tensor * op);
    void (*prepare)(ggml_compute_params * params, const ggml_tensor * op);
    size_t (*get_n_step)();
    void (*run)(const ggml_tensor * op, const void * workspace, size_t n_start, size_t n_end);
};

struct kernel_registry {
    const kernel * const * entries;
    size_t                 size;
};

struct packed_row_decoder {
    const uint8_t * b;
    const uint8_t * qparams;
    size_t          k;
    size_t          n;
    size_t          experts;
    size_t          rlenb;
    size_t          segment_size;
    size_t          block_size;
    bool            asymmetric;
};

struct weight_packer {
    const char * name;
    bool (*supports)(const ggml_tensor * tensor);
    size_t (*get_alloc_size)(const ggml_tensor * tensor);
    bool (*pack)(ggml_tensor * tensor, const void * data);
    bool (*is_packed)(const ggml_tensor * tensor);
    bool (*open_decoder)(const ggml_tensor * tensor, packed_row_decoder & decoder);
    void (*decode_row)(const packed_row_decoder & decoder, int32_t expert_idx, int32_t row_idx, float * dst);
};

struct weight_packer_registry {
    const weight_packer * const * entries;
    size_t                        size;
};

const kernel_registry &        get_kernel_registry();
const weight_packer_registry & get_weight_packer_registry();

}  // namespace ggml::cpu::torq_tile
