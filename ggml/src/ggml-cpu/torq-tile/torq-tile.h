// SPDX-FileCopyrightText: Copyright 2026 C-SKY Microsystems Co., Ltd.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "ggml-alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_torq_tile_buffer_type(void);

#ifdef __cplusplus
}
#endif
