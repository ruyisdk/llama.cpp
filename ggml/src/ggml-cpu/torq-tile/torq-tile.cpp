// SPDX-FileCopyrightText: Copyright 2026 C-SKY Microsystems Co., Ltd.
// SPDX-License-Identifier: MIT
//

#include "torq-tile.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "kernels.h"
#include "traits.h"
#include "vec.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {

bool torq_tile_is_disabled() {
    // Read once so packed weights and operation dispatch use the same setting.
    static const bool disabled = []() {
        const char * value = std::getenv("GGML_TORQ_TILE_DISABLE");
        if (value == nullptr) {
            return false;
        }
        const bool off = value[0] == '\0' || std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
                         std::strcmp(value, "no") == 0 || std::strcmp(value, "off") == 0;
        if (off) {
            return false;
        }
        GGML_LOG_INFO("torq_tile_is_disabled: GGML_TORQ_TILE_DISABLE=%s, using generic CPU kernels\n", value);
        return true;
    }();

    return disabled;
}

bool is_host(const ggml_tensor * tensor) {
    return tensor->buffer == nullptr || ggml_backend_buft_is_host(tensor->buffer->buft) ||
           tensor->buffer->buft == ggml_backend_cpu_torq_tile_buffer_type();
}

bool supports_mul_mat_shape(const ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    if (ggml_get_op_params_i32(op, 1) != GGML_HINT_NONE) {
        return false;
    }

    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || op->ne[2] != 1 ||
        op->ne[3] != 1) {
        return false;
    }

    if (src0->ne[0] != src1->ne[0] || op->ne[0] != src0->ne[1] || op->ne[1] != src1->ne[1]) {
        return false;
    }
    if (src0->ne[0] == 0) {
        return false;
    }

    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return false;
    }

    if (!is_host(src0) || !is_host(src1) || !is_host(op)) {
        return false;
    }

    return true;
}

bool supports_packed_mul_mat_fallback_shape(const ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    if (ggml_get_op_params_i32(op, 1) != GGML_HINT_NONE || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->ne[0] <= 0 || src0->ne[0] > std::numeric_limits<int>::max() || src0->ne[1] <= 0 ||
        src0->ne[1] > std::numeric_limits<int32_t>::max() || src0->ne[2] <= 0 ||
        src0->ne[2] > std::numeric_limits<int32_t>::max() || src0->ne[3] != 1) {
        return false;
    }

    if (src1->ne[0] != src0->ne[0] || src1->ne[1] <= 0 || src1->ne[2] <= 0 || src1->ne[3] <= 0 ||
        src1->ne[2] % src0->ne[2] != 0 || op->ne[0] != src0->ne[1] || op->ne[1] != src1->ne[1] ||
        op->ne[2] != src1->ne[2] || op->ne[3] != src1->ne[3]) {
        return false;
    }

    if (src1->nb[0] != sizeof(float) || op->nb[0] != sizeof(float)) {
        return false;
    }

    return is_host(src0) && is_host(src1) && is_host(op);
}

const ggml::cpu::torq_tile::kernel * select_mul_mat_kernel(const ggml_tensor * op) {
    if (!supports_mul_mat_shape(op)) {
        return nullptr;
    }

    const auto & registry = ggml::cpu::torq_tile::get_kernel_registry();
    for (size_t i = 0; i < registry.size; ++i) {
        const ggml::cpu::torq_tile::kernel * candidate = registry.entries[i];
        if (candidate->supports(op)) {
            return candidate;
        }
    }

    return nullptr;
}

const ggml::cpu::torq_tile::weight_packer * select_weight_packer(const ggml_tensor * tensor) {
    if (torq_tile_is_disabled()) {
        return nullptr;
    }

    const auto & registry = ggml::cpu::torq_tile::get_weight_packer_registry();
    for (size_t i = 0; i < registry.size; ++i) {
        const ggml::cpu::torq_tile::weight_packer * candidate = registry.entries[i];
        if (candidate->supports(tensor)) {
            return candidate;
        }
    }
    return nullptr;
}

struct packed_weight_source {
    const ggml::cpu::torq_tile::weight_packer * packer  = nullptr;
    const ggml_tensor *                         storage = nullptr;
};

packed_weight_source find_packed_weight_source(const ggml_tensor * tensor) {
    const ggml_tensor * storage = tensor;
    while (storage->view_src != nullptr) {
        storage = storage->view_src;
    }

    const ggml::cpu::torq_tile::weight_packer * packer = select_weight_packer(storage);
    return packer != nullptr && packer->is_packed(storage) ? packed_weight_source{ packer, storage } :
                                                             packed_weight_source{};
}

const ggml::cpu::torq_tile::weight_packer * select_packed_weight_packer(const ggml_tensor * tensor) {
    return find_packed_weight_source(tensor).packer;
}

bool supports_packed_row_view(const ggml_tensor * tensor, const ggml_tensor * storage) {
    if (storage == nullptr) {
        return false;
    }
    if (tensor == storage) {
        return true;
    }

    for (const ggml_tensor * view = tensor; view != storage; view = view->view_src) {
        if (view == nullptr || view->view_src == nullptr || view->view_offs != 0) {
            return false;
        }
    }

    return tensor->type == storage->type && tensor->ne[0] == storage->ne[0] && tensor->ne[3] == 1 &&
           storage->ne[3] == 1 && ggml_is_contiguous(tensor) && ggml_nelements(tensor) == ggml_nelements(storage);
}

bool supports_packed_row_cast(const ggml_tensor * op, const ggml_tensor * storage, size_t & row) {
    if (op->op != GGML_OP_CPY || op->src[0] == nullptr || storage == nullptr) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    if (src0->type != storage->type || src0->ne[0] != storage->ne[0] || storage->ne[2] != 1 || storage->ne[3] != 1 ||
        !ggml_is_vector(src0) || !ggml_is_contiguous(src0) || op->type != GGML_TYPE_F32 || op->ne[0] != src0->ne[0] ||
        !ggml_is_vector(op) || !ggml_is_contiguous(op) || !is_host(op)) {
        return false;
    }

    size_t offset = 0;
    for (const ggml_tensor * view = src0; view != storage; view = view->view_src) {
        if (view == nullptr || view->view_src == nullptr ||
            view->view_offs > std::numeric_limits<size_t>::max() - offset) {
            return false;
        }
        offset += view->view_offs;
    }

    // View offsets use raw GGML row sizes, not packed byte offsets.
    const size_t row_size = ggml_row_size(storage->type, storage->ne[0]);
    if (row_size == 0 || offset % row_size != 0) {
        return false;
    }
    row = offset / row_size;
    return row < static_cast<size_t>(storage->ne[1]);
}

void map_packed_row(const ggml_tensor *                              tensor,
                    const ggml::cpu::torq_tile::packed_row_decoder & decoder,
                    int32_t                                          expert_idx,
                    int32_t                                          row_idx,
                    int32_t &                                        storage_expert_idx,
                    int32_t &                                        storage_row_idx) {
    GGML_ASSERT(expert_idx >= 0 && expert_idx < tensor->ne[2]);
    GGML_ASSERT(row_idx >= 0 && row_idx < tensor->ne[1]);

    const size_t flat_row =
        static_cast<size_t>(expert_idx) * static_cast<size_t>(tensor->ne[1]) + static_cast<size_t>(row_idx);
    const size_t storage_expert = flat_row / decoder.n;
    const size_t storage_row    = flat_row % decoder.n;
    GGML_ASSERT(storage_expert < decoder.experts);
    GGML_ASSERT(storage_expert <= static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    GGML_ASSERT(storage_row <= static_cast<size_t>(std::numeric_limits<int32_t>::max()));

    storage_expert_idx = static_cast<int32_t>(storage_expert);
    storage_row_idx    = static_cast<int32_t>(storage_row);
}

bool supports_get_rows_shape(const ggml_tensor * op) {
    return op->op == GGML_OP_GET_ROWS && op->src[0] != nullptr && op->src[1] != nullptr &&
           op->src[1]->type == GGML_TYPE_I32 && op->type == GGML_TYPE_F32 && is_host(op->src[1]) && is_host(op);
}

size_t split_start(size_t count, size_t part, size_t parts) {
    return (count / parts) * part + std::min(count % parts, part);
}

size_t packed_mul_mat_fallback_work_size(const ggml_tensor * op, int n_threads) {
    const size_t k   = static_cast<size_t>(op->src[0]->ne[0]);
    const size_t nth = static_cast<size_t>(std::max(n_threads, 1));
    if (k > std::numeric_limits<size_t>::max() / sizeof(float) / nth) {
        GGML_ABORT("%s: workspace size overflow for tensor '%s'\n", __func__, op->src[0]->name);
    }
    return nth * k * sizeof(float);
}

void compute_packed_mul_mat_fallback(ggml_compute_params *        params,
                                     ggml_tensor *                op,
                                     const packed_weight_source & source) {
    GGML_ASSERT(supports_packed_mul_mat_fallback_shape(op));
    GGML_ASSERT(supports_packed_row_view(op->src[0], source.storage));

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    ggml::cpu::torq_tile::packed_row_decoder decoder = {};
    if (!source.packer->open_decoder(source.storage, decoder)) {
        GGML_ABORT("%s: cannot decode packed tensor '%s'\n", __func__, source.storage->name);
    }

    const size_t k       = static_cast<size_t>(src0->ne[0]);
    const size_t n       = static_cast<size_t>(src0->ne[1]);
    const size_t experts = static_cast<size_t>(src0->ne[2]);
    GGML_ASSERT(decoder.k == k && decoder.n * decoder.experts == n * experts);
    GGML_ASSERT(params->wdata != nullptr && params->wsize >= packed_mul_mat_fallback_work_size(op, params->nth));

    const size_t nth     = static_cast<size_t>(std::max(params->nth, 1));
    const size_t ith     = static_cast<size_t>(params->ith);
    const size_t n_start = split_start(n, ith, nth);
    const size_t n_end   = split_start(n, ith + 1, nth);
    float *      row     = static_cast<float *>(params->wdata) + ith * k;

    const int64_t batch_ratio = src1->ne[2] / src0->ne[2];
    for (size_t output_row = n_start; output_row < n_end; ++output_row) {
        int32_t decoded_expert = -1;
        for (int64_t i12 = 0; i12 < src1->ne[2]; ++i12) {
            const int32_t expert = static_cast<int32_t>(i12 / batch_ratio);
            if (expert != decoded_expert) {
                int32_t storage_expert = 0;
                int32_t storage_row    = 0;
                map_packed_row(src0, decoder, expert, static_cast<int32_t>(output_row), storage_expert, storage_row);
                source.packer->decode_row(decoder, storage_expert, storage_row, row);
                decoded_expert = expert;
            }

            for (int64_t i13 = 0; i13 < src1->ne[3]; ++i13) {
                for (int64_t i11 = 0; i11 < src1->ne[1]; ++i11) {
                    const auto * input =
                        reinterpret_cast<const float *>(static_cast<const uint8_t *>(src1->data) + i11 * src1->nb[1] +
                                                        i12 * src1->nb[2] + i13 * src1->nb[3]);
                    auto * output =
                        reinterpret_cast<float *>(static_cast<uint8_t *>(op->data) + output_row * op->nb[0] +
                                                  i11 * op->nb[1] + i12 * op->nb[2] + i13 * op->nb[3]);
                    ggml_vec_dot_f32(static_cast<int>(k), output, 0, row, 0, input, 0, 1);
                }
            }
        }
    }
}

class tensor_traits final : public ggml::cpu::tensor_traits {
    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) override {
        const packed_weight_source packed_source =
            op->src[0] == nullptr ? packed_weight_source{} : find_packed_weight_source(op->src[0]);
        const ggml::cpu::torq_tile::weight_packer * packed_packer = packed_source.packer;
        if (packed_packer != nullptr && op->op != GGML_OP_GET_ROWS && op->op != GGML_OP_MUL_MAT) {
            size = 0;
            return true;
        }

        if (packed_packer != nullptr && !supports_packed_row_view(op->src[0], packed_source.storage)) {
            size = 0;
            return true;
        }

        if (supports_get_rows_shape(op) && select_packed_weight_packer(op->src[0]) != nullptr) {
            size = 0;
            return true;
        }

        const ggml::cpu::torq_tile::kernel * kernel = select_mul_mat_kernel(op);
        if (kernel == nullptr) {
            if (packed_packer != nullptr) {
                size =
                    supports_packed_mul_mat_fallback_shape(op) ? packed_mul_mat_fallback_work_size(op, n_threads) : 0;
                return true;
            }
            return false;
        }

        size = kernel->get_work_size == nullptr ? 0 : kernel->get_work_size(op);
        return true;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        const packed_weight_source packed_source =
            op->src[0] == nullptr ? packed_weight_source{} : find_packed_weight_source(op->src[0]);
        const ggml::cpu::torq_tile::weight_packer * packed_packer = packed_source.packer;
        if (packed_packer != nullptr && op->op == GGML_OP_CPY) {
            size_t row = 0;
            GGML_ASSERT(supports_packed_row_cast(op, packed_source.storage, row));
            if (params->ith == 0) {
                ggml::cpu::torq_tile::packed_row_decoder decoder = {};
                if (!packed_packer->open_decoder(packed_source.storage, decoder)) {
                    GGML_ABORT("%s: cannot decode packed tensor '%s'\n", __func__, packed_source.storage->name);
                }
                packed_packer->decode_row(decoder, 0, static_cast<int32_t>(row), static_cast<float *>(op->data));
            }
            return true;
        }
        if (packed_packer != nullptr && op->op != GGML_OP_GET_ROWS && op->op != GGML_OP_MUL_MAT) {
            GGML_ABORT("%s: packed tensor '%s' cannot be used by operation %s\n", __func__, op->src[0]->name,
                       ggml_op_name(op->op));
        }

        if (packed_packer != nullptr && !supports_packed_row_view(op->src[0], packed_source.storage)) {
            GGML_ABORT("%s: packed tensor '%s' uses an unsupported view of '%s'\n", __func__, op->src[0]->name,
                       packed_source.storage->name);
        }

        if (supports_get_rows_shape(op)) {
            const ggml::cpu::torq_tile::weight_packer * packer = packed_packer;
            if (packer == nullptr) {
                return false;
            }

            const ggml_tensor *                      src0    = op->src[0];
            const ggml_tensor *                      src1    = op->src[1];
            ggml::cpu::torq_tile::packed_row_decoder decoder = {};
            if (!packer->open_decoder(packed_source.storage, decoder)) {
                GGML_ABORT("%s: cannot decode packed tensor '%s'\n", __func__, packed_source.storage->name);
            }
            GGML_ASSERT(decoder.k == static_cast<size_t>(src0->ne[0]));
            GGML_ASSERT(decoder.n * decoder.experts ==
                        static_cast<size_t>(src0->ne[1]) * static_cast<size_t>(src0->ne[2]));

            const int64_t nr             = ggml_nelements(src1);
            const int64_t nth            = std::max(params->nth, 1);
            const int64_t dr             = (nr + nth - 1) / nth;
            const int64_t ir0            = dr * params->ith;
            const int64_t ir1            = std::min(ir0 + dr, nr);
            const int64_t rows_per_batch = src1->ne[0] * src1->ne[1];

            for (int64_t i = ir0; i < ir1; ++i) {
                const int64_t i12 = i / rows_per_batch;
                const int64_t i11 = (i - i12 * rows_per_batch) / src1->ne[0];
                const int64_t i10 = i - i12 * rows_per_batch - i11 * src1->ne[0];

                const auto * index = static_cast<const uint8_t *>(src1->data) + i10 * src1->nb[0] + i11 * src1->nb[1] +
                                     i12 * src1->nb[2];
                const int32_t row_idx = *reinterpret_cast<const int32_t *>(index);
                GGML_ASSERT(row_idx >= 0 && row_idx < src0->ne[1]);

                auto *        dst = reinterpret_cast<float *>(static_cast<uint8_t *>(op->data) + i10 * op->nb[1] +
                                                              i11 * op->nb[2] + i12 * op->nb[3]);
                const int32_t expert_idx         = static_cast<int32_t>(i11 + i12 * src0->ne[2]);
                int32_t       storage_expert_idx = 0;
                int32_t       storage_row_idx    = 0;
                map_packed_row(src0, decoder, expert_idx, row_idx, storage_expert_idx, storage_row_idx);
                packer->decode_row(decoder, storage_expert_idx, storage_row_idx, dst);
            }

            return true;
        }

        const ggml::cpu::torq_tile::kernel * kernel = select_mul_mat_kernel(op);
        if (kernel == nullptr) {
            if (packed_packer != nullptr && supports_packed_mul_mat_fallback_shape(op)) {
                compute_packed_mul_mat_fallback(params, op, packed_source);
                return true;
            }
            if (packed_packer != nullptr) {
                GGML_ABORT("%s: no Ztt path can consume packed tensor '%s' for operation %s\n", __func__,
                           op->src[0]->name, ggml_op_name(op->op));
            }
            return false;
        }
        if (kernel->prepare != nullptr) {
            kernel->prepare(params, op);
            ggml_barrier(params->threadpool);
        }

        const ggml_tensor * src0 = op->src[0];
        const size_t        n    = static_cast<size_t>(src0->ne[1]);

        const size_t n_step   = kernel->get_n_step();
        // Each thread owns complete packed N tiles, including the final tail.
        const size_t n_blocks = (n + n_step - 1) / n_step;
        const size_t nth      = static_cast<size_t>(std::max(params->nth, 1));
        const size_t ith      = static_cast<size_t>(params->ith);

        const size_t block_start = split_start(n_blocks, ith, nth);
        const size_t block_end   = split_start(n_blocks, ith + 1, nth);
        const size_t n_start     = block_start * n_step;
        const size_t n_end       = std::min(n, block_end * n_step);

        if (n_start >= n_end) {
            return true;
        }

        kernel->run(op, params->wdata, n_start, n_end);

        return true;
    }
};

tensor_traits * get_tensor_traits_instance() {
    static tensor_traits traits;
    return &traits;
}

class extra_buffer_type final : public ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) override {
        if (torq_tile_is_disabled()) {
            return false;
        }

        if (op->src[0] == nullptr || op->src[0]->buffer == nullptr ||
            op->src[0]->buffer->buft != ggml_backend_cpu_torq_tile_buffer_type()) {
            return false;
        }

        const ggml_tensor * storage = op->src[0];
        while (storage->view_src != nullptr) {
            storage = storage->view_src;
        }
        if (op->op == GGML_OP_CPY) {
            size_t row = 0;
            return select_weight_packer(storage) != nullptr && supports_packed_row_cast(op, storage, row);
        }
        if (!supports_packed_row_view(op->src[0], storage)) {
            return false;
        }

        if (op->op == GGML_OP_GET_ROWS) {
            if (!supports_get_rows_shape(op)) {
                return false;
            }
            return select_weight_packer(op->src[0]) != nullptr;
        }

        if (op->op == GGML_OP_MUL_MAT && op->src[1] != nullptr && is_host(op->src[1]) &&
            op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
            select_weight_packer(op->src[0]) != nullptr) {
            return supports_mul_mat_shape(op) || supports_packed_mul_mat_fallback_shape(op);
        }

        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {
        if (torq_tile_is_disabled()) {
            return nullptr;
        }
        if (op->src[0] != nullptr && op->src[0]->buffer != nullptr &&
            op->src[0]->buffer->buft == ggml_backend_cpu_torq_tile_buffer_type() && op->src[0]->extra != nullptr &&
            select_packed_weight_packer(op->src[0]) != nullptr) {
            return static_cast<ggml::cpu::tensor_traits *>(op->src[0]->extra);
        }

        if (op->op == GGML_OP_CPY && op->src[0] != nullptr) {
            const packed_weight_source source = find_packed_weight_source(op->src[0]);
            size_t                     row    = 0;
            if (source.packer != nullptr && supports_packed_row_cast(op, source.storage, row)) {
                return get_tensor_traits_instance();
            }
        }

        if (supports_get_rows_shape(op) && select_packed_weight_packer(op->src[0]) != nullptr) {
            return get_tensor_traits_instance();
        }

        return select_mul_mat_kernel(op) != nullptr ? get_tensor_traits_instance() : nullptr;
    }
};

const char * buffer_type_name(ggml_backend_buffer_type_t) {
    return "CPU_TORQ_TILE";
}

enum ggml_status buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor * tensor) {
    tensor->extra = get_tensor_traits_instance();
    return GGML_STATUS_SUCCESS;
}

bool packed_rows_match_reference(const ggml_tensor *                         tensor,
                                 const void *                                data,
                                 const ggml::cpu::torq_tile::weight_packer * packer) {
    const ggml_to_float_t dequantize_row = ggml_get_type_traits(tensor->type)->to_float;
    if (dequantize_row == nullptr) {
        return false;
    }

    ggml::cpu::torq_tile::packed_row_decoder decoder = {};
    if (!packer->open_decoder(tensor, decoder)) {
        return false;
    }

    const int64_t k        = tensor->ne[0];
    const int64_t n        = tensor->ne[1];
    const int64_t experts  = tensor->ne[2];
    const size_t  row_size = ggml_row_size(tensor->type, k);
    const int64_t samples  = std::min<int64_t>(n, experts == 1 ? 8 : 3);
    try {
        std::vector<float> reference(k);
        std::vector<float> decoded(k);

        for (int64_t expert = 0; expert < experts; ++expert) {
            for (int64_t sample = 0; sample < samples; ++sample) {
                const int64_t row = samples == 1 ? 0 : (n - 1) * sample / (samples - 1);

                const size_t raw_row = static_cast<size_t>(expert * n + row);
                const auto * raw     = static_cast<const uint8_t *>(data) + raw_row * row_size;
                dequantize_row(raw, reference.data(), k);
                packer->decode_row(decoder, static_cast<int32_t>(expert), static_cast<int32_t>(row), decoded.data());

                float absmax = 0.0f;
                for (int64_t i = 0; i < k; ++i) {
                    absmax = std::max(absmax, std::fabs(reference[i]));
                }

                const float tolerance = 0.01f * absmax + 1e-6f;
                for (int64_t i = 0; i < k; ++i) {
                    if (!(std::fabs(decoded[i] - reference[i]) <= tolerance)) {
                        return false;
                    }
                }
            }
        }
    } catch (const std::bad_alloc &) {
        return false;
    }

    return true;
}

void buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    const ggml::cpu::torq_tile::weight_packer * packer = select_weight_packer(tensor);
    if (packer == nullptr) {
        std::memcpy(tensor->data, data, size);
        return;
    }

    const char * failure = nullptr;
    if (!packer->pack(tensor, data)) {
        failure = "conversion failed";
    } else if (!packed_rows_match_reference(tensor, data, packer)) {
        failure = "decoded rows do not match the reference dequantization";
    }

    if (failure != nullptr) {
        GGML_ABORT("%s: cannot pack tensor '%s' with %s (%s)\n", __func__, tensor->name, packer->name, failure);
    }
}

void buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (select_packed_weight_packer(tensor) != nullptr) {
        GGML_ABORT("%s: packed tensor '%s' has no raw GGML representation for readback, copy, or save\n", __func__,
                   tensor->name);
    }
    std::memcpy(data, static_cast<const uint8_t *>(tensor->data) + offset, size);
}

ggml_backend_buffer_t buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft                = buft;
    buffer->iface.init_tensor   = buffer_init_tensor;
    buffer->iface.set_tensor    = buffer_set_tensor;
    buffer->iface.get_tensor    = buffer_get_tensor;
    buffer->iface.set_tensor_2d = nullptr;
    buffer->iface.get_tensor_2d = nullptr;
    buffer->iface.cpy_tensor    = nullptr;
    return buffer;
}

size_t buffer_type_alignment(ggml_backend_buffer_type_t) {
    return TENSOR_ALIGNMENT;
}

size_t buffer_type_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) {
    const ggml::cpu::torq_tile::weight_packer * packer = select_weight_packer(tensor);
    return packer == nullptr ? ggml_nbytes(tensor) : packer->get_alloc_size(tensor);
}

}  // namespace

ggml_backend_buffer_type_t ggml_backend_cpu_torq_tile_buffer_type(void) {
    static extra_buffer_type        context;
    static ggml_backend_buffer_type buffer_type = {
        /* .iface = */ {
                        /* .get_name         = */ buffer_type_name,
                        /* .alloc_buffer     = */ buffer_type_alloc_buffer,
                        /* .alloc_buffer_n  = */ nullptr,
                        /* .get_alignment    = */ buffer_type_alignment,
                        /* .get_max_size     = */ nullptr,
                        /* .get_alloc_size   = */ buffer_type_alloc_size,
                        /* .get_alloc_size_n = */ nullptr,
                        /* .is_host          = */ nullptr,
                        },
        /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ &context,
    };

    return &buffer_type;
}
