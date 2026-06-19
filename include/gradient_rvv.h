#pragma once

#include <cstdint>
#include <cstdlib>
#include <riscv_vector.h>

// =============================================================================
// Phase 6.5 — RVV implementation of Sobel L1 magnitude
//
// Computes:
//      magnitude = |Gx| + |Gy|
//
// Then normalizes output to [0, 255], same as the scalar function.
//
// This version vectorizes the hot L1 magnitude computation:
// - load Gx and Gy as int16 vectors
// - compute absolute values using max(x, -x)
// - add |Gx| + |Gy|
// - store raw uint16 magnitude
//
// The normalization pass is kept scalar for this first correct RVV version.
// =============================================================================

inline void gradient_magnitude_l1_rvv(
    const int16_t* __restrict__ Gx,
    const int16_t* __restrict__ Gy,
    uint8_t* __restrict__ mag_out,
    int width,
    int height)
{
    int n = width * height;

    // Worst case |Gx| + |Gy| is about 4080, so uint16_t is enough.
    uint16_t* raw = new uint16_t[n];

    // -------------------------------------------------------------------------
    // Pass 1: RVV compute raw magnitude = |Gx| + |Gy|
    // -------------------------------------------------------------------------
    for (int i = 0; i < n; ) {
        // Vector-length agnostic strip-mining.
        // vl depends on the hardware/QEMU VLEN and remaining elements.
        size_t vl = __riscv_vsetvl_e16m1(n - i);

        // Load consecutive Gx and Gy values.
        // LMUL=m1 is enough here because we only keep a few vector registers live.
        vint16m1_t vx = __riscv_vle16_v_i16m1(&Gx[i], vl);
        vint16m1_t vy = __riscv_vle16_v_i16m1(&Gy[i], vl);

        // Compute -Gx and -Gy using reverse subtract: 0 - value.
        vint16m1_t neg_x = __riscv_vrsub_vx_i16m1(vx, 0, vl);
        vint16m1_t neg_y = __riscv_vrsub_vx_i16m1(vy, 0, vl);

        // Absolute value using max(value, -value).
        vint16m1_t abs_x_i16 = __riscv_vmax_vv_i16m1(vx, neg_x, vl);
        vint16m1_t abs_y_i16 = __riscv_vmax_vv_i16m1(vy, neg_y, vl);

        // Reinterpret positive int16 values as uint16.
        vuint16m1_t abs_x = __riscv_vreinterpret_v_i16m1_u16m1(abs_x_i16);
        vuint16m1_t abs_y = __riscv_vreinterpret_v_i16m1_u16m1(abs_y_i16);

        // Add |Gx| + |Gy|.
        vuint16m1_t mag = __riscv_vadd_vv_u16m1(abs_x, abs_y, vl);

        // Store raw unnormalized magnitude.
        __riscv_vse16_v_u16m1(&raw[i], mag, vl);

        i += vl;
    }

    // -------------------------------------------------------------------------
    // Pass 2: scalar max search
    // -------------------------------------------------------------------------
    uint16_t max_mag = 0;
    for (int i = 0; i < n; ++i) {
        if (raw[i] > max_mag) {
            max_mag = raw[i];
        }
    }

    // -------------------------------------------------------------------------
    // Pass 3: scalar normalization to [0, 255]
    // -------------------------------------------------------------------------
    if (max_mag == 0) {
        for (int i = 0; i < n; ++i) {
            mag_out[i] = 0;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            mag_out[i] = static_cast<uint8_t>((raw[i] * 255) / max_mag);
        }
    }

    delete[] raw;
}
