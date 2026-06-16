#pragma once
// =============================================================================
// gradient.h
// Phase 2 — Sections 2.3, 2.4, 2.5
// Sobel gradient computation, gradient magnitude (L1 & L2), gradient direction.
//
// Design notes:
//   - Gx and Gy are stored as SEPARATE arrays (Structure of Arrays / SoA).
//     This lets a future RVV implementation load consecutive Gx values with
//     a single vector load instruction.  Interleaved AoS would need gather ops.
//   - All public functions take raw pointers + dimensions so they work without
//     any STL container and compile cleanly with the RISC-V bare-metal toolchain.
//   - Boundary handling: zero-padding (same as the Gaussian stage).
// =============================================================================

#include <cstdint>
#include <cmath>
#include <cstring>   // memset
#include <cstdlib>   // abs (integer)
#include <algorithm> // std::max, std::min

// ---------------------------------------------------------------------------
// Direction constants (quantized to 4 angles)
// ---------------------------------------------------------------------------
static constexpr uint8_t DIR_0   = 0;   //   0° — horizontal gradient
static constexpr uint8_t DIR_45  = 1;   //  45° — diagonal
static constexpr uint8_t DIR_90  = 2;   //  90° — vertical gradient
static constexpr uint8_t DIR_135 = 3;   // 135° — anti-diagonal

// ===========================================================================
// 2.3  Sobel gradient computation
// ===========================================================================
// Applies Sobel-X and Sobel-Y 3×3 kernels to a blurred uint8_t image.
// Outputs are stored into caller-allocated int16_t arrays (Gx, Gy).
//
// Why int16_t is sufficient:
//   Sobel-X worst case = 4 * 255 - 4*(-255) = 4*510 = 2040  (<32767)  ✓
//
// Sobel-X kernel:  Sobel-Y kernel:
//  -1  0  1         -1 -2 -1
//  -2  0  2          0  0  0
//  -1  0  1          1  2  1
// ===========================================================================

inline void compute_sobel(
    const uint8_t* __restrict__ blurred,  // input: blurred grayscale image
    int16_t*       __restrict__ Gx,       // output: horizontal gradient
    int16_t*       __restrict__ Gy,       // output: vertical gradient
    int width,
    int height)
{
    // Sobel kernels stored as flat 3×3 row-major arrays
    // (row, col) → index = row*3 + col
    static const int8_t kx[9] = {
        -1,  0,  1,
        -2,  0,  2,
        -1,  0,  1
    };
    static const int8_t ky[9] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int32_t sum_x = 0;
            int32_t sum_y = 0;

            // Iterate over the 3×3 kernel
            for (int kr = -1; kr <= 1; ++kr) {
                for (int kc = -1; kc <= 1; ++kc) {
                    int nr = r + kr;
                    int nc = c + kc;

                    // Zero-padding: pixels outside image boundaries treated as 0
                    uint8_t pixel = 0;
                    if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
                        pixel = blurred[nr * width + nc];
                    }

                    int kid = (kr + 1) * 3 + (kc + 1);  // kernel flat index
                    sum_x += kx[kid] * pixel;
                    sum_y += ky[kid] * pixel;
                }
            }

            Gx[r * width + c] = static_cast<int16_t>(sum_x);
            Gy[r * width + c] = static_cast<int16_t>(sum_y);
        }
    }
}

// ===========================================================================
// 2.4  Gradient magnitude
// ===========================================================================
// Two implementations; output is normalized to [0, 255] and written into
// a caller-allocated uint8_t array.
//
// Normalization requires two passes:
//   Pass 1 — find the maximum magnitude across the whole image.
//   Pass 2 — scale every pixel:  out[i] = (mag[i] * 255) / max_mag
// A single-pass approach is not straightforward because you don't know the
// global max until you've seen all pixels.
// ===========================================================================

// --- L1 norm: |Gx| + |Gy|  (integer-only, fast, slight over-estimate) ------
inline void gradient_magnitude_l1(
    const int16_t* __restrict__ Gx,
    const int16_t* __restrict__ Gy,
    uint8_t*       __restrict__ mag_out,
    int width,
    int height)
{
    int n = width * height;

    // Temporary buffer for raw (un-normalised) magnitudes
    // Worst case: |Gx|+|Gy| ≤ 2040 + 2040 = 4080 — fits comfortably in int32_t
    int32_t* raw = new int32_t[n];

    // Pass 1: compute raw magnitudes and find max
    int32_t max_mag = 0;
    for (int i = 0; i < n; ++i) {
        raw[i] = static_cast<int32_t>(std::abs((int)Gx[i]))
               + static_cast<int32_t>(std::abs((int)Gy[i]));
        if (raw[i] > max_mag) max_mag = raw[i];
    }

    // Pass 2: normalise to [0, 255]
    if (max_mag == 0) {
        // Uniform image — all zeros
        for (int i = 0; i < n; ++i) mag_out[i] = 0;
    } else {
        for (int i = 0; i < n; ++i) {
            mag_out[i] = static_cast<uint8_t>((raw[i] * 255) / max_mag);
        }
    }

    delete[] raw;
}

// --- L2 norm: sqrt(Gx² + Gy²)  (mathematically exact, uses float sqrt) ----
inline void gradient_magnitude_l2(
    const int16_t* __restrict__ Gx,
    const int16_t* __restrict__ Gy,
    uint8_t*       __restrict__ mag_out,
    int width,
    int height)
{
    int n = width * height;

    float* raw = new float[n];

    // Pass 1: compute raw magnitudes
    float max_mag = 0.0f;
    for (int i = 0; i < n; ++i) {
        float gx = static_cast<float>(Gx[i]);
        float gy = static_cast<float>(Gy[i]);
        raw[i] = std::sqrt(gx * gx + gy * gy);
        if (raw[i] > max_mag) max_mag = raw[i];
    }

    // Pass 2: normalise
    if (max_mag < 1e-6f) {
        for (int i = 0; i < n; ++i) mag_out[i] = 0;
    } else {
        for (int i = 0; i < n; ++i) {
            mag_out[i] = static_cast<uint8_t>((raw[i] / max_mag) * 255.0f);
        }
    }

    delete[] raw;
}

// ===========================================================================
// 2.5  Gradient direction (quantized)
// ===========================================================================
// Outputs one of {0, 1, 2, 3} for each pixel representing 0°/45°/90°/135°.
//
// We avoid floating-point atan2 by using integer cross-multiplication:
//
//   Let ax = |Gx|,  ay = |Gy|
//   Boundary angles: tan(22.5°) ≈ 2/5   tan(67.5°) ≈ 12/5
//
//   if  ay*5 <  ax*2   → angle < 22.5°   → DIR_0   (nearly horizontal)
//   if  ay*5 >= ax*12  → angle > 67.5°   → DIR_90  (nearly vertical)
//   else depends on sign of Gx*Gy        → DIR_45 or DIR_135
//
// This is a standard embedded trick to avoid fp division in a hot loop.
// ===========================================================================
inline void gradient_direction(
    const int16_t* __restrict__ Gx,
    const int16_t* __restrict__ Gy,
    uint8_t*       __restrict__ dir_out,
    int width,
    int height)
{
    int n = width * height;

    for (int i = 0; i < n; ++i) {
        int32_t ax = std::abs((int32_t)Gx[i]);
        int32_t ay = std::abs((int32_t)Gy[i]);

        // Use integer cross-multiplication to avoid division / atan2
        // Multiply up by 5 to keep everything integer
        if (ay * 5 < ax * 2) {
            // |angle| < 22.5°  →  horizontal gradient  →  DIR_0
            dir_out[i] = DIR_0;
        } else if (ay * 5 >= ax * 12) {
            // |angle| > 67.5°  →  vertical gradient  →  DIR_90
            dir_out[i] = DIR_90;
        } else {
            // 22.5° ≤ |angle| ≤ 67.5°  →  diagonal
            // Sign of Gx*Gy distinguishes 45° from 135°
            //   Gx > 0, Gy > 0  → gradient points up-right  → edge is at 135°
            //   Gx > 0, Gy < 0  → gradient points down-right → edge is at 45°
            //   (and symmetric for negative Gx)
            if ((Gx[i] > 0) == (Gy[i] > 0)) {
                dir_out[i] = DIR_135;
            } else {
                dir_out[i] = DIR_45;
            }
        }
    }
}
