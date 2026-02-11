/* Copyright 2023-2024 Apple Inc. */

#ifndef MLX_HALF_H
#define MLX_HALF_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// Parse-time aliases for ffigen. MLX uses architecture-native fp16/bf16,
// which are 16-bit storage types in ABI.
#ifndef HAS_FLOAT16
#define HAS_FLOAT16
#endif
typedef uint16_t float16_t;

#ifndef HAS_BFLOAT16
#define HAS_BFLOAT16
#endif
typedef uint16_t bfloat16_t;

#ifdef __cplusplus
}
#endif

#endif
