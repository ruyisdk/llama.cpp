// SPDX-FileCopyrightText: Copyright 2026 C-SKY Microsystems Co., Ltd.
// SPDX-License-Identifier: MIT
//

#define GGML_COMMON_DECL_CPP

#include "kernels.h"

#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "ggml-impl.h"
#include "src/gemm/gemm_i8_i8/gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt.h"
#include "src/gemm/gemm_i8_u4/gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p/tqt_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt.h"
#include "torq-tile.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace ggml::cpu::torq_tile {
namespace {

constexpr uint32_t packed_weight_magic          = 0x34545154;
constexpr uint32_t packed_weight_version        = 4;
constexpr uint32_t packed_weight_qau4c32_ztt_id = 21;
constexpr uint32_t packed_weight_qsi8c16_ztt_id = 22;
constexpr size_t   packed_weight_alignment      = 64;
static_assert((packed_weight_alignment & (packed_weight_alignment - 1)) == 0,
              "packed_weight_alignment must be a power of two");

using get_packed_size_fn  = size_t (*)(size_t, size_t);
using get_layout_width_fn = size_t (*)(void);
using run_pack_fn         = void (*)(size_t, size_t, size_t, size_t, size_t, const void *, void *);

struct packed_weight_variant {
    enum ggml_type      type;
    uint32_t            kernel_id;
    size_t              logical_b_divisor;
    size_t              qparam_block_size;
    bool                asymmetric;
    get_layout_width_fn get_layout_width;
    get_packed_size_fn  get_b_packed_size;
    get_packed_size_fn  get_qparams_packed_size;
};

struct packed_weight_header {
    uint32_t magic;
    uint32_t version;
    uint32_t kernel_id;
    uint32_t original_type;
    uint32_t rlenb;
    uint32_t reserved;
    uint64_t raw_size;
    uint64_t b_offset;
    uint64_t b_size;
    uint64_t b_stride;
    uint64_t qparams_offset;
    uint64_t qparams_size;
    uint64_t qparams_stride;
    uint64_t total_size;
    uint64_t n;
    uint64_t k;
    uint64_t experts;
};

struct packed_weight_layout {
    size_t logical_b_size;
    size_t b_offset;
    size_t b_size;
    size_t b_stride;
    size_t qparams_offset;
    size_t qparams_size;
    size_t qparams_stride;
    size_t total_size;
};

bool checked_add(size_t a, size_t b, size_t & result);
bool checked_multiply(size_t a, size_t b, size_t & result);
bool checked_align(size_t value, size_t alignment, size_t & result);
bool get_packed_weight_layout(const ggml_tensor *           tensor,
                              const packed_weight_variant & variant,
                              packed_weight_layout &        layout);
void write_packed_weight_header(ggml_tensor *                 tensor,
                                const packed_weight_variant & variant,
                                const packed_weight_layout &  layout);

struct flat_qparams_layout {
    size_t scale_offset;
    size_t real_zero_offset;
    size_t total_size;
};

bool get_flat_qparams_layout(size_t n, size_t k, size_t block_size, bool asymmetric, flat_qparams_layout & layout) {
    size_t count = 0;
    size_t bytes = 0;
    if (!checked_multiply(n, k / block_size, count) || !checked_multiply(count, sizeof(float), bytes)) {
        return false;
    }

    size_t cursor       = 0;
    layout.scale_offset = cursor;
    if (!checked_add(cursor, bytes, cursor)) {
        return false;
    }

    layout.real_zero_offset = 0;
    if (asymmetric) {
        layout.real_zero_offset = cursor;
        if (!checked_add(cursor, bytes, cursor)) {
            return false;
        }
    }

    layout.total_size = cursor;
    return true;
}

size_t get_flat_qparams_size(size_t n, size_t k, size_t block_size, bool asymmetric) {
    flat_qparams_layout layout = {};
    return get_flat_qparams_layout(n, k, block_size, asymmetric, layout) ? layout.total_size : 0;
}

size_t get_qau4c32_flat_qparams_size(size_t n, size_t k) {
    return get_flat_qparams_size(n, k, 32, true);
}

size_t get_qsi6c16_flat_qparams_size(size_t n, size_t k) {
    return get_flat_qparams_size(n, k, 16, false);
}

const packed_weight_variant q4_k_qau4c32_ztt_weight = {
    /* .type                    = */ GGML_TYPE_Q4_K,
    /* .kernel_id               = */ packed_weight_qau4c32_ztt_id,
    /* .logical_b_divisor       = */ 2,
    /* .qparam_block_size       = */ 32,
    /* .asymmetric              = */ true,
    /* .get_layout_width        = */ tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt,
    /* .get_b_packed_size       = */
    tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt,
    /* .get_qparams_packed_size = */ get_qau4c32_flat_qparams_size,
};

const packed_weight_variant q6_k_qsi8c16_ztt_weight = {
    /* .type                    = */ GGML_TYPE_Q6_K,
    /* .kernel_id               = */ packed_weight_qsi8c16_ztt_id,
    /* .logical_b_divisor       = */ 1,
    /* .qparam_block_size       = */ 16,
    /* .asymmetric              = */ false,
    /* .get_layout_width        = */ tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    /* .get_b_packed_size       = */
    tqt_get_b_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    /* .get_qparams_packed_size = */ get_qsi6c16_flat_qparams_size,
};

bool checked_add(size_t a, size_t b, size_t & result) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return false;
    }
    result = a + b;
    return true;
}

bool checked_multiply(size_t a, size_t b, size_t & result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool checked_align(size_t value, size_t alignment, size_t & result) {
    size_t padded = 0;
    if (!checked_add(value, alignment - 1, padded)) {
        return false;
    }
    result = padded & ~(alignment - 1);
    return true;
}

bool supports_packed_layout_width(size_t width) {
    return width != 0 && width <= 128 && (width & (width - 1)) == 0;
}

bool supports_packed_weight(const ggml_tensor * tensor, const packed_weight_variant & variant) {
    return tensor->type == variant.type && tensor->ne[0] > 0 && tensor->ne[1] > 0 &&
           tensor->ne[0] <= std::numeric_limits<int>::max() && tensor->ne[1] <= std::numeric_limits<int32_t>::max() &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1 && static_cast<size_t>(tensor->ne[0]) % QK_K == 0 &&
           ggml_is_contiguous(tensor) && supports_packed_layout_width(variant.get_layout_width());
}

bool get_packed_weight_layout(const ggml_tensor *           tensor,
                              const packed_weight_variant & variant,
                              packed_weight_layout &        layout) {
    if (!supports_packed_weight(tensor, variant)) {
        return false;
    }

    const size_t n            = static_cast<size_t>(tensor->ne[1]);
    const size_t k            = static_cast<size_t>(tensor->ne[0]);
    const size_t experts      = static_cast<size_t>(tensor->ne[2]);
    const size_t layout_width = variant.get_layout_width();
    if (!supports_packed_layout_width(layout_width)) {
        return false;
    }

    size_t element_count = 0;
    if (!checked_multiply(n, k, element_count) || element_count % variant.logical_b_divisor != 0) {
        return false;
    }
    layout.logical_b_size = element_count / variant.logical_b_divisor;

    const size_t b_size_per_expert       = variant.get_b_packed_size(n, k);
    const size_t qparams_size_per_expert = variant.get_qparams_packed_size(n, k);

    if (!checked_align(b_size_per_expert, packed_weight_alignment, layout.b_stride) ||
        !checked_align(qparams_size_per_expert, packed_weight_alignment, layout.qparams_stride)) {
        return false;
    }

    size_t b_prefix       = 0;
    size_t qparams_prefix = 0;
    if (!checked_multiply(experts - 1, layout.b_stride, b_prefix) ||
        !checked_add(b_prefix, b_size_per_expert, layout.b_size) ||
        !checked_multiply(experts - 1, layout.qparams_stride, qparams_prefix) ||
        !checked_add(qparams_prefix, qparams_size_per_expert, layout.qparams_size)) {
        return false;
    }

    size_t cursor = sizeof(packed_weight_header);
    if (!checked_align(cursor, packed_weight_alignment, layout.b_offset) ||
        !checked_add(layout.b_offset, layout.b_size, cursor) ||
        !checked_align(cursor, packed_weight_alignment, layout.qparams_offset) ||
        !checked_add(layout.qparams_offset, layout.qparams_size, layout.total_size)) {
        return false;
    }

    return true;
}

size_t get_packed_weight_alloc_size(const ggml_tensor * tensor, const packed_weight_variant & variant) {
    packed_weight_layout layout = {};
    return get_packed_weight_layout(tensor, variant, layout) ? std::max(layout.total_size, ggml_nbytes(tensor)) :
                                                               ggml_nbytes(tensor);
}

void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

void write_packed_weight_header(ggml_tensor *                 tensor,
                                const packed_weight_variant & variant,
                                const packed_weight_layout &  layout) {
    const packed_weight_header complete = {
        /* .magic          = */ packed_weight_magic,
        /* .version        = */ packed_weight_version,
        /* .kernel_id      = */ variant.kernel_id,
        /* .original_type  = */ static_cast<uint32_t>(variant.type),
        /* .rlenb          = */ static_cast<uint32_t>(variant.get_layout_width()),
        /* .reserved       = */ 0,
        /* .raw_size       = */ ggml_nbytes(tensor),
        /* .b_offset       = */ layout.b_offset,
        /* .b_size         = */ layout.b_size,
        /* .b_stride       = */ layout.b_stride,
        /* .qparams_offset = */ layout.qparams_offset,
        /* .qparams_size   = */ layout.qparams_size,
        /* .qparams_stride = */ layout.qparams_stride,
        /* .total_size     = */ layout.total_size,
        /* .n              = */ static_cast<uint64_t>(tensor->ne[1]),
        /* .k              = */ static_cast<uint64_t>(tensor->ne[0]),
        /* .experts        = */ static_cast<uint64_t>(tensor->ne[2]),
    };
    auto * base   = static_cast<uint8_t *>(tensor->data);
    auto * header = reinterpret_cast<packed_weight_header *>(base);
    std::memcpy(header, &complete, sizeof(complete));
}

using build_flat_row_fn = bool (*)(const void * raw, size_t k, uint8_t * codes, float * scales, float * real_zeros);

bool pack_flat(ggml_tensor *                 tensor,
               const void *                  data,
               const packed_weight_variant & variant,
               build_flat_row_fn             build_row,
               run_pack_fn                   run_b_pack) {
    packed_weight_layout layout = {};
    if (!get_packed_weight_layout(tensor, variant, layout)) {
        return false;
    }

    const size_t n = static_cast<size_t>(tensor->ne[1]);
    const size_t k = static_cast<size_t>(tensor->ne[0]);

    flat_qparams_layout qparams_layout = {};
    if (!get_flat_qparams_layout(n, k, variant.qparam_block_size, variant.asymmetric, qparams_layout)) {
        return false;
    }

    auto * base    = static_cast<uint8_t *>(tensor->data);
    auto * qparams = base + layout.qparams_offset;
    auto * scales  = reinterpret_cast<float *>(qparams + qparams_layout.scale_offset);
    auto * real_zeros =
        variant.asymmetric ? reinterpret_cast<float *>(qparams + qparams_layout.real_zero_offset) : nullptr;

    const size_t row_size = ggml_row_size(variant.type, static_cast<int64_t>(k));
    const size_t blocks   = k / variant.qparam_block_size;

    try {
        std::vector<uint8_t> logical_b(layout.logical_b_size, 0);
        const size_t         logical_stride = layout.logical_b_size / n;

        for (size_t row = 0; row < n; ++row) {
            if (!build_row(static_cast<const uint8_t *>(data) + row * row_size, k,
                           logical_b.data() + row * logical_stride, scales + row * blocks,
                           real_zeros == nullptr ? nullptr : real_zeros + row * blocks)) {
                return false;
            }
        }

        std::memset(base, 0, sizeof(packed_weight_header));
        run_b_pack(n, k, k, k, 0, logical_b.data(), base + layout.b_offset);
    } catch (const std::bad_alloc &) {
        return false;
    }

    write_packed_weight_header(tensor, variant, layout);
    return true;
}

bool build_q4_k_qau4c32_row(const void * raw, size_t k, uint8_t * codes, float * scales, float * real_zeros) {
    // GGML pairs codes at j and j + 32; the kernel expects adjacent K codes.
    const auto * blocks = static_cast<const block_q4_K *>(raw);
    for (size_t block_idx = 0; block_idx < k / QK_K; ++block_idx) {
        const block_q4_K & block = blocks[block_idx];
        const float        d     = GGML_FP16_TO_FP32(block.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        const float        dmin  = GGML_FP16_TO_FP32(block.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
        if (!std::isfinite(d) || !std::isfinite(dmin)) {
            return false;
        }

        for (size_t subblock = 0; subblock < 8; ++subblock) {
            const uint8_t * quants = block.qs + (subblock / 2) * 32;
            const bool      high   = subblock % 2 != 0;
            uint8_t *       packed = codes + (block_idx * 8 + subblock) * 16;

            for (size_t pair = 0; pair < 16; ++pair) {
                const uint8_t q0 = high ? quants[2 * pair] >> 4 : quants[2 * pair] & 0x0f;
                const uint8_t q1 = high ? quants[2 * pair + 1] >> 4 : quants[2 * pair + 1] & 0x0f;
                packed[pair]     = q0 | (q1 << 4);
            }

            uint8_t scale = 0;
            uint8_t min   = 0;
            get_scale_min_k4(static_cast<int>(subblock), block.scales, scale, min);
            const size_t qparam_idx = block_idx * 8 + subblock;
            scales[qparam_idx]      = d * scale;
            real_zeros[qparam_idx]  = dmin * min;
        }
    }
    return true;
}

bool build_q6_k_qsi8c16_ztt_row(const void * raw, size_t k, uint8_t * codes, float * scales, float *) {
    const auto * blocks = static_cast<const block_q6_K *>(raw);
    for (size_t b = 0; b < k / QK_K; ++b) {
        const auto & block = blocks[b];
        const float  d     = GGML_FP16_TO_FP32(block.d);
        if (!std::isfinite(d)) {
            return false;
        }
        auto * dst = reinterpret_cast<int8_t *>(codes) + b * QK_K;
        for (size_t half = 0; half < 2; ++half) {
            const auto * ql = block.ql + half * 64;
            const auto * qh = block.qh + half * 32;
            for (size_t l = 0; l < 32; ++l) {
                dst[half * 128 + l]      = ((ql[l] & 15) | ((qh[l] & 3) << 4)) - 32;
                dst[half * 128 + l + 32] = ((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32;
                dst[half * 128 + l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                dst[half * 128 + l + 96] = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
        }
        for (size_t g = 0; g < QK_K / 16; ++g) {
            scales[b * (QK_K / 16) + g] = d * block.scales[g];
        }
    }
    return true;
}

bool pack_q4_k_ztt(ggml_tensor * tensor, const void * data) {
    return pack_flat(tensor, data, q4_k_qau4c32_ztt_weight, build_q4_k_qau4c32_row,
                     tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt);
}

bool pack_q6_k_ztt(ggml_tensor * tensor, const void * data) {
    return pack_flat(tensor, data, q6_k_qsi8c16_ztt_weight, build_q6_k_qsi8c16_ztt_row,
                     tqt_run_bt_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt);
}

const packed_weight_header * get_packed_weight_header(const ggml_tensor *           tensor,
                                                      const packed_weight_variant & variant) {
    if (tensor->data == nullptr || tensor->buffer == nullptr ||
        ggml_backend_buffer_get_type(tensor->buffer) != ggml_backend_cpu_torq_tile_buffer_type()) {
        return nullptr;
    }
    packed_weight_layout layout = {};
    if (!get_packed_weight_layout(tensor, variant, layout)) {
        return nullptr;
    }

    const auto * header = static_cast<const packed_weight_header *>(tensor->data);
    if (header->magic != packed_weight_magic || header->version != packed_weight_version ||
        header->kernel_id != variant.kernel_id || header->original_type != static_cast<uint32_t>(variant.type) ||
        header->rlenb != static_cast<uint32_t>(variant.get_layout_width()) || header->raw_size != ggml_nbytes(tensor) ||
        header->b_offset != layout.b_offset || header->b_size != layout.b_size || header->b_stride != layout.b_stride ||
        header->qparams_offset != layout.qparams_offset || header->qparams_size != layout.qparams_size ||
        header->qparams_stride != layout.qparams_stride || header->total_size != layout.total_size ||
        header->n != static_cast<uint64_t>(tensor->ne[1]) || header->k != static_cast<uint64_t>(tensor->ne[0]) ||
        header->experts != static_cast<uint64_t>(tensor->ne[2])) {
        return nullptr;
    }

    return header;
}

bool open_packed_decoder(const ggml_tensor *           tensor,
                         const packed_weight_variant & variant,
                         packed_row_decoder &          decoder) {
    const packed_weight_header * header = get_packed_weight_header(tensor, variant);
    if (header == nullptr) {
        return false;
    }

    const auto * base     = static_cast<const uint8_t *>(tensor->data);
    decoder.b             = base + header->b_offset;
    decoder.qparams       = base + header->qparams_offset;
    decoder.k             = static_cast<size_t>(header->k);
    decoder.n             = static_cast<size_t>(header->n);
    decoder.experts       = static_cast<size_t>(header->experts);
    decoder.rlenb         = header->rlenb;
    decoder.block_size    = variant.qparam_block_size;
    decoder.asymmetric    = variant.asymmetric;
    const size_t padded_n = (decoder.n + decoder.rlenb - 1) / decoder.rlenb * decoder.rlenb;
    decoder.segment_size  = header->b_size / padded_n;
    return true;
}

template <const packed_weight_variant & variant> struct packed_weight_ops {
    static bool supports(const ggml_tensor * tensor) {
        packed_weight_layout layout = {};
        return get_packed_weight_layout(tensor, variant, layout);
    }

    static size_t get_alloc_size(const ggml_tensor * tensor) { return get_packed_weight_alloc_size(tensor, variant); }

    static bool is_packed(const ggml_tensor * tensor) { return get_packed_weight_header(tensor, variant) != nullptr; }

    static bool open_decoder(const ggml_tensor * tensor, packed_row_decoder & decoder) {
        return open_packed_decoder(tensor, variant, decoder);
    }
};

void decode_ztt_row(const packed_row_decoder & decoder, int32_t expert_idx, int32_t row_idx, float * dst) {
    GGML_ASSERT(expert_idx == 0);
    GGML_ASSERT(row_idx >= 0 && static_cast<size_t>(row_idx) < decoder.n);
    const size_t row    = static_cast<size_t>(row_idx);
    const size_t step   = decoder.rlenb;
    const size_t groups = decoder.k / decoder.block_size;
    const auto * scales = reinterpret_cast<const float *>(decoder.qparams);
    const auto * zeros  = decoder.asymmetric ? scales + decoder.n * groups : nullptr;
    const size_t panel  = row / step * step * decoder.segment_size;
    for (size_t q = 0; q < decoder.k; ++q) {
        const size_t index = (q / step) * step * step + (row % step) * step + q % step;
        int          code;
        if (decoder.asymmetric) {
            // Undo the packer's unsigned-to-signed nibble conversion.
            code = ((decoder.b[panel + index / 2] >> ((index % 2) * 4)) & 15) ^ 8;
        } else {
            code = static_cast<int8_t>(decoder.b[panel + index]);
        }
        const size_t group = row * groups + q / decoder.block_size;
        dst[q]             = code * scales[group] - (zeros ? zeros[group] : 0.0f);
    }
}

using pack_fn       = bool (*)(ggml_tensor *, const void *);
using decode_row_fn = void (*)(const packed_row_decoder &, int32_t, int32_t, float *);

template <const packed_weight_variant & variant>
constexpr weight_packer make_weight_packer(const char * name, pack_fn pack, decode_row_fn decode) {
    return {
        /* .name                = */ name,
        /* .supports            = */ packed_weight_ops<variant>::supports,
        /* .get_alloc_size      = */ packed_weight_ops<variant>::get_alloc_size,
        /* .pack                = */ pack,
        /* .is_packed           = */ packed_weight_ops<variant>::is_packed,
        /* .open_decoder        = */ packed_weight_ops<variant>::open_decoder,
        /* .decode_row          = */ decode,
    };
}

const weight_packer q4_k_ztt_packer =
    make_weight_packer<q4_k_qau4c32_ztt_weight>("q4_k_to_qau4c32_ztt", pack_q4_k_ztt, decode_ztt_row);
const weight_packer q6_k_ztt_packer =
    make_weight_packer<q6_k_qsi8c16_ztt_weight>("q6_k_to_qsi8c16_ztt", pack_q6_k_ztt, decode_ztt_row);

struct qsi8d256_activation_layout {
    size_t packed_a_offset;
    size_t codes_offset;
    size_t scales_offset;
    size_t bsums_offset;
    size_t temporary_offset;
    size_t total_size;
};

bool get_qsi8d256_activation_layout(size_t                       m,
                                    size_t                       k,
                                    bool                         bsums,
                                    size_t                       packed_a_size,
                                    qsi8d256_activation_layout & layout) {
    size_t codes_size     = 0;
    size_t scales_count   = 0;
    size_t scales_size    = 0;
    size_t bsums_count    = 0;
    size_t bsums_size     = 0;
    size_t temporary_size = 0;
    if (!checked_multiply(m, k, codes_size) || !checked_multiply(m, k / QK_K, scales_count) ||
        !checked_multiply(scales_count, sizeof(float), scales_size) ||
        !checked_multiply(k / QK_K, sizeof(block_q8_K), temporary_size)) {
        return false;
    }
    if (bsums &&
        (!checked_multiply(m, k / 16, bsums_count) || !checked_multiply(bsums_count, sizeof(int16_t), bsums_size))) {
        return false;
    }

    size_t cursor = 0;
    if (!checked_align(cursor, packed_weight_alignment, layout.packed_a_offset) ||
        !checked_add(layout.packed_a_offset, packed_a_size, cursor) ||
        !checked_align(cursor, packed_weight_alignment, layout.codes_offset) ||
        !checked_add(layout.codes_offset, codes_size, cursor) ||
        !checked_align(cursor, alignof(float), layout.scales_offset) ||
        !checked_add(layout.scales_offset, scales_size, cursor)) {
        return false;
    }

    layout.bsums_offset = 0;
    if (bsums) {
        if (!checked_align(cursor, alignof(int16_t), layout.bsums_offset) ||
            !checked_add(layout.bsums_offset, bsums_size, cursor)) {
            return false;
        }
    }

    if (!checked_align(cursor, alignof(block_q8_K), layout.temporary_offset) ||
        !checked_add(layout.temporary_offset, temporary_size, layout.total_size)) {
        return false;
    }
    return true;
}

void prepare_qsi8d256_activations(ggml_compute_params * params,
                                  const ggml_tensor *   op,
                                  bool                  bsums,
                                  get_packed_size_fn    get_a_packed_size,
                                  run_pack_fn           run_a_pack) {
    if (params->ith != 0) {
        return;
    }

    const size_t               m             = static_cast<size_t>(op->src[1]->ne[1]);
    const size_t               k             = static_cast<size_t>(op->src[0]->ne[0]);
    qsi8d256_activation_layout layout        = {};
    const size_t               packed_a_size = get_a_packed_size(m, k);
    GGML_ASSERT(get_qsi8d256_activation_layout(m, k, bsums, packed_a_size, layout));
    GGML_ASSERT(params->wdata != nullptr && params->wsize >= layout.total_size);

    auto *       workspace   = static_cast<uint8_t *>(params->wdata);
    auto *       packed_a    = workspace + layout.packed_a_offset;
    auto *       codes       = reinterpret_cast<int8_t *>(workspace + layout.codes_offset);
    auto *       scales      = reinterpret_cast<float *>(workspace + layout.scales_offset);
    auto *       sums        = bsums ? reinterpret_cast<int16_t *>(workspace + layout.bsums_offset) : nullptr;
    auto *       temporary   = reinterpret_cast<block_q8_K *>(workspace + layout.temporary_offset);
    const auto * activations = static_cast<const float *>(op->src[1]->data);
    const size_t blocks      = k / QK_K;

    for (size_t row = 0; row < m; ++row) {
        quantize_row_q8_K(activations + row * k, temporary, k);
        for (size_t block = 0; block < blocks; ++block) {
            const block_q8_K & quant = temporary[block];
            std::memcpy(codes + row * k + block * QK_K, quant.qs, QK_K);
            scales[row * blocks + block] = quant.d;
            if (sums != nullptr) {
                // Recompute sums because the scalar quantizer leaves zero-block bsums unset.
                for (size_t group = 0; group < QK_K / 16; ++group) {
                    int sum = 0;
                    for (size_t i = 0; i < 16; ++i) {
                        sum += quant.qs[group * 16 + i];
                    }
                    sums[row * (k / 16) + block * (QK_K / 16) + group] = static_cast<int16_t>(sum);
                }
            }
        }
    }

    run_a_pack(m, k, k, k, 0, codes, packed_a);
}

using run_flat_gemm_fn = void (*)(size_t,
                                  size_t,
                                  size_t,
                                  const void *,
                                  const void *,
                                  void *,
                                  size_t,
                                  const float *,
                                  size_t,
                                  const int16_t *,
                                  size_t,
                                  const float *,
                                  const float *,
                                  size_t);

struct flat_kernel_variant {
    const packed_weight_variant * weight;
    bool                          bsums;
    get_packed_size_fn            get_a_packed_size;
    run_pack_fn                   run_a_pack;
    size_t (*get_n_step)();
    size_t (*get_b_packed_offset)(size_t, size_t, size_t, size_t);
    run_flat_gemm_fn run_gemm;
};

template <typename Params,
          void Run(size_t,
                   size_t,
                   size_t,
                   const void *,
                   size_t,
                   size_t,
                   const void *,
                   size_t,
                   size_t,
                   const void *,
                   size_t,
                   void *,
                   size_t,
                   const void *,
                   float,
                   float,
                   const Params *)>
void run_flat_asymmetric_gemm(size_t          m,
                              size_t          n,
                              size_t          k,
                              const void *    a,
                              const void *    b,
                              void *          d,
                              size_t          ldd,
                              const float *   scale_a,
                              size_t          ldsa,
                              const int16_t * bsums_a,
                              size_t          ldbsuma,
                              const float *   scale_b,
                              const float *   real_zero_b,
                              size_t          ldsb) {
    const Params params = {
        /* .scale_a     = */ scale_a,
        /* .ldsa        = */ ldsa,
        /* .bsums_a     = */ bsums_a,
        /* .ldbsuma     = */ ldbsuma,
        /* .scale_b     = */ scale_b,
        /* .real_zero_b = */ real_zero_b,
        /* .ldsb        = */ ldsb,
    };
    Run(m, n, k, a, k, 0, b, k, 0, nullptr, ldd, d, ldd, nullptr, -INFINITY, INFINITY, &params);
}

template <typename Params,
          void Run(size_t,
                   size_t,
                   size_t,
                   const void *,
                   size_t,
                   size_t,
                   const void *,
                   size_t,
                   size_t,
                   const void *,
                   size_t,
                   void *,
                   size_t,
                   const void *,
                   float,
                   float,
                   const Params *)>
void run_flat_symmetric_gemm(size_t        m,
                             size_t        n,
                             size_t        k,
                             const void *  a,
                             const void *  b,
                             void *        d,
                             size_t        ldd,
                             const float * scale_a,
                             size_t        ldsa,
                             const int16_t *,
                             size_t,
                             const float * scale_b,
                             const float *,
                             size_t ldsb) {
    const Params params = {
        /* .scale_a = */ scale_a,
        /* .ldsa    = */ ldsa,
        /* .scale_b = */ scale_b,
        /* .ldsb    = */ ldsb,
    };
    Run(m, n, k, a, k, 0, b, k, 0, nullptr, ldd, d, ldd, nullptr, -INFINITY, INFINITY, &params);
}

const flat_kernel_variant q4_k_ztt_kernel_variant = {
    /* .weight              = */ &q4_k_qau4c32_ztt_weight,
    /* .bsums               = */ true,
    /* .get_a_packed_size   = */ tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt,
    /* .run_a_pack          = */ tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt,
    /* .get_n_step          = */ tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt,
    /* .get_b_packed_offset = */ tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt,
    /* .run_gemm            = */
    run_flat_asymmetric_gemm<tqt_qsi8d256_qau4c32_params,
                             tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qau4c32p_1nlenx1nlen_ztt>,
};

const flat_kernel_variant q6_k_ztt_kernel_variant = {
    /* .weight              = */ &q6_k_qsi8c16_ztt_weight,
    /* .bsums               = */ false,
    /* .get_a_packed_size   = */ tqt_get_a_packed_size_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    /* .run_a_pack          = */ tqt_run_a_pack_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    /* .get_n_step          = */ tqt_get_n_step_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    /* .get_b_packed_offset = */ tqt_get_b_packed_offset_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt,
    /* .run_gemm            = */
    run_flat_symmetric_gemm<tqt_qsi8d256_qsi8c16_params,
                            tqt_run_gemm_1xnbias_clamp_f32_qsi8d256p_qsi8c16p_1nlenx1nlen_ztt>,
};

struct flat_activation_offsets {
    size_t packed_a_offset;
    size_t scales_offset;
    size_t bsums_offset;
    size_t scale_leading_dimension;
    size_t total_size;
};

bool get_flat_activation_offsets(size_t                      m,
                                 size_t                      k,
                                 const flat_kernel_variant & variant,
                                 flat_activation_offsets &   offsets) {
    qsi8d256_activation_layout layout = {};
    if (!get_qsi8d256_activation_layout(m, k, variant.bsums, variant.get_a_packed_size(m, k), layout)) {
        return false;
    }
    offsets.packed_a_offset         = layout.packed_a_offset;
    offsets.scales_offset           = layout.scales_offset;
    offsets.bsums_offset            = layout.bsums_offset;
    offsets.scale_leading_dimension = k / QK_K;
    offsets.total_size              = layout.total_size;
    return true;
}

bool supports_flat_f32(const ggml_tensor * op, const flat_kernel_variant & variant) {
    return op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && op->src[1]->ne[1] > 0 &&
           get_packed_weight_header(op->src[0], *variant.weight) != nullptr;
}

size_t get_work_size_flat_f32(const ggml_tensor * op, const flat_kernel_variant & variant) {
    const size_t            m       = static_cast<size_t>(op->src[1]->ne[1]);
    const size_t            k       = static_cast<size_t>(op->src[0]->ne[0]);
    flat_activation_offsets offsets = {};
    GGML_ASSERT(get_flat_activation_offsets(m, k, variant, offsets));
    return offsets.total_size;
}

void prepare_flat_f32(ggml_compute_params * params, const ggml_tensor * op, const flat_kernel_variant & variant) {
    prepare_qsi8d256_activations(params, op, variant.bsums, variant.get_a_packed_size, variant.run_a_pack);
}

void run_flat_f32(const ggml_tensor *         op,
                  const void *                workspace,
                  size_t                      n_start,
                  size_t                      n_end,
                  const flat_kernel_variant & variant) {
    const ggml_tensor * src0   = op->src[0];
    const auto *        header = get_packed_weight_header(src0, *variant.weight);
    GGML_ASSERT(header != nullptr);

    const size_t m          = static_cast<size_t>(op->src[1]->ne[1]);
    const size_t n          = static_cast<size_t>(src0->ne[1]);
    const size_t k          = static_cast<size_t>(src0->ne[0]);
    const size_t block_size = variant.weight->qparam_block_size;

    flat_activation_offsets offsets = {};
    GGML_ASSERT(get_flat_activation_offsets(m, k, variant, offsets));

    flat_qparams_layout qparams_layout = {};
    GGML_ASSERT(get_flat_qparams_layout(n, k, block_size, variant.weight->asymmetric, qparams_layout));

    const auto * workspace_bytes = static_cast<const uint8_t *>(workspace);
    const auto * base            = static_cast<const uint8_t *>(src0->data);
    const auto * qparams         = base + header->qparams_offset;
    const auto * b               = base + header->b_offset + variant.get_b_packed_offset(n_start, 0, k, n);
    const auto * scale_a         = reinterpret_cast<const float *>(workspace_bytes + offsets.scales_offset);
    const auto * bsums_a =
        variant.bsums ? reinterpret_cast<const int16_t *>(workspace_bytes + offsets.bsums_offset) : nullptr;
    const auto * scale_b =
        reinterpret_cast<const float *>(qparams + qparams_layout.scale_offset) + n_start * (k / block_size);
    const auto * real_zero_b =
        variant.weight->asymmetric ?
            reinterpret_cast<const float *>(qparams + qparams_layout.real_zero_offset) + n_start * (k / block_size) :
            nullptr;
    auto * d = static_cast<uint8_t *>(op->data) + n_start * op->nb[0];

    variant.run_gemm(m, n_end - n_start, k, workspace_bytes + offsets.packed_a_offset, b, d, op->nb[1] / sizeof(float),
                     scale_a, offsets.scale_leading_dimension, bsums_a, k / 16, scale_b, real_zero_b, k / block_size);
}

template <const flat_kernel_variant & variant> struct flat_kernel_ops {
    static bool supports(const ggml_tensor * op) { return supports_flat_f32(op, variant); }

    static size_t get_work_size(const ggml_tensor * op) { return get_work_size_flat_f32(op, variant); }

    static void prepare(ggml_compute_params * params, const ggml_tensor * op) { prepare_flat_f32(params, op, variant); }

    static size_t get_n_step() { return variant.get_n_step(); }

    static void run(const ggml_tensor * op, const void * workspace, size_t n_start, size_t n_end) {
        run_flat_f32(op, workspace, n_start, n_end, variant);
    }
};

template <const flat_kernel_variant & variant> constexpr kernel make_flat_kernel() {
    return {
        /* .supports      = */ flat_kernel_ops<variant>::supports,
        /* .get_work_size = */ flat_kernel_ops<variant>::get_work_size,
        /* .prepare       = */ flat_kernel_ops<variant>::prepare,
        /* .get_n_step    = */ flat_kernel_ops<variant>::get_n_step,
        /* .run           = */ flat_kernel_ops<variant>::run,
    };
}

const kernel q4_k_ztt_kernel = make_flat_kernel<q4_k_ztt_kernel_variant>();
const kernel q6_k_ztt_kernel = make_flat_kernel<q6_k_ztt_kernel_variant>();

}  // namespace

const kernel_registry & get_kernel_registry() {
    static const kernel *        entries[] = { &q4_k_ztt_kernel, &q6_k_ztt_kernel };
    static const kernel_registry registry  = { entries, 2 };
    return registry;
}

const weight_packer_registry & get_weight_packer_registry() {
    static const weight_packer *        entries[] = { &q4_k_ztt_packer, &q6_k_ztt_packer };
    static const weight_packer_registry registry  = { entries, 2 };
    return registry;
}

}  // namespace ggml::cpu::torq_tile
