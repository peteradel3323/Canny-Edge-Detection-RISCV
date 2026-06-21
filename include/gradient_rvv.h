#pragma once

#include <cstdint>
#include <cstdlib>
#include <riscv_vector.h>

// =============================================================================
// Phase 6.5 — Improved RVV implementation of Sobel L1 magnitude
//
// Computes:
//      magnitude = |Gx| + |Gy|
//
// Improvements in this version:
// - RVV computes raw magnitude.
// - RVV widens |Gx| and |Gy| from u16 to u32 before addition.
// - RVV reduction finds max magnitude.
// - RVV normalization computes (raw * 255) / max_mag using u32 vectors.
// - Final cast to uint8_t is kept scalar for portability and correctness.
// =============================================================================

inline void gradient_magnitude_l1_rvv(
    const int16_t* __restrict__ Gx,
    const int16_t* __restrict__ Gy,
    uint8_t* __restrict__ mag_out,
    int width,
    int height)
{
    const int n = width * height;

    // raw stores unnormalized magnitude values as u32.
    // u32 is used because normalization later does raw * 255.
    uint32_t* raw = new uint32_t[n];
    uint32_t* norm32 = new uint32_t[n];

    // -------------------------------------------------------------------------
    // Pass 1: RVV raw magnitude calculation
    // raw = |Gx| + |Gy|
    // -------------------------------------------------------------------------
    for (int i = 0; i < n; ) {
        // Set active vector length for 16-bit elements.
        // This makes the loop vector-length agnostic.
        size_t vl = __riscv_vsetvl_e16m1(n - i);

        // Load Gx and Gy as signed 16-bit vectors.
        vint16m1_t vx = __riscv_vle16_v_i16m1(&Gx[i], vl);
        vint16m1_t vy = __riscv_vle16_v_i16m1(&Gy[i], vl);

        // Compute negative versions: 0 - x.
        vint16m1_t neg_x = __riscv_vrsub_vx_i16m1(vx, 0, vl);
        vint16m1_t neg_y = __riscv_vrsub_vx_i16m1(vy, 0, vl);

        // Absolute value using max(x, -x).
        vint16m1_t abs_x_i16 = __riscv_vmax_vv_i16m1(vx, neg_x, vl);
        vint16m1_t abs_y_i16 = __riscv_vmax_vv_i16m1(vy, neg_y, vl);

        // Reinterpret non-negative signed values as unsigned 16-bit values.
        vuint16m1_t abs_x_u16 = __riscv_vreinterpret_v_i16m1_u16m1(abs_x_i16);
        vuint16m1_t abs_y_u16 = __riscv_vreinterpret_v_i16m1_u16m1(abs_y_i16);

        // Widen u16 -> u32 before adding.
        // This prevents overflow and prepares for normalization.
        vuint32m2_t abs_x_u32 = __riscv_vzext_vf2_u32m2(abs_x_u16, vl);
        vuint32m2_t abs_y_u32 = __riscv_vzext_vf2_u32m2(abs_y_u16, vl);

        // Raw L1 magnitude: |Gx| + |Gy|.
        vuint32m2_t mag_u32 = __riscv_vadd_vv_u32m2(abs_x_u32, abs_y_u32, vl);

        // Store raw unnormalized magnitude.
        __riscv_vse32_v_u32m2(&raw[i], mag_u32, vl);

        i += vl;
    }

    // -------------------------------------------------------------------------
    // Pass 2: RVV reduction to find max_mag
    // -------------------------------------------------------------------------
    uint32_t init_value[1] = {0};
    size_t vl_one = __riscv_vsetvl_e32m1(1);

    // Reduction accumulator vector. Element 0 holds the current max.
    vuint32m1_t max_vec = __riscv_vle32_v_u32m1(init_value, vl_one);

    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m2(n - i);

        // Load raw u32 magnitudes.
        vuint32m2_t raw_vec = __riscv_vle32_v_u32m2(&raw[i], vl);

        // Reduce current vector into max_vec.
        max_vec = __riscv_vredmaxu_vs_u32m2_u32m1(raw_vec, max_vec, vl);

        i += vl;
    }

    // Extract max value by storing element 0 of max_vec.
    uint32_t max_store[1] = {0};
    __riscv_vse32_v_u32m1(max_store, max_vec, vl_one);
    uint32_t max_mag = max_store[0];

    // -------------------------------------------------------------------------
    // Pass 3: RVV normalization
    // norm32 = (raw * 255) / max_mag
    // -------------------------------------------------------------------------
    if (max_mag == 0) {
        for (int i = 0; i < n; ++i) {
            mag_out[i] = 0;
        }

        delete[] raw;
        delete[] norm32;
        return;
    }

    for (int i = 0; i < n; ) {
        size_t vl = __riscv_vsetvl_e32m2(n - i);

        // Load raw u32 magnitude.
        vuint32m2_t raw_vec = __riscv_vle32_v_u32m2(&raw[i], vl);

        // Widening is already handled by storing raw as u32.
        // Multiply by 255 safely in u32.
        vuint32m2_t scaled_vec = __riscv_vmul_vx_u32m2(raw_vec, 255u, vl);

        // Divide by max_mag to normalize to [0, 255].
        vuint32m2_t norm_vec = __riscv_vdivu_vx_u32m2(scaled_vec, max_mag, vl);

        // Store normalized u32 result.
        __riscv_vse32_v_u32m2(&norm32[i], norm_vec, vl);

        i += vl;
    }

    // -------------------------------------------------------------------------
    // Pass 4: final scalar cast u32 -> u8
    // Values are already in [0, 255].
    // -------------------------------------------------------------------------
    for (int i = 0; i < n; ++i) {
        mag_out[i] = static_cast<uint8_t>(norm32[i]);
    }

    delete[] raw;
    delete[] norm32;
}
