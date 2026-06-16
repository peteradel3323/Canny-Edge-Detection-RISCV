// =============================================================================
// test_gradient.cpp
// Phase 3 — Unit tests for gradient.h  (Sobel, magnitude, direction)
//
// Build and run natively on the host machine with GoogleTest:
//   g++ -std=c++17 -I../include test_gradient.cpp -lgtest -lgtest_main -pthread -o test_gradient
//   ./test_gradient
//
// These tests are host-side only.  They do NOT use the RISC-V cross-compiler.
// QEMU-side equivalence tests are in test_gradient_qemu.cpp.
// =============================================================================

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "gradient.h"

// ---------------------------------------------------------------------------
// Helper: allocate aligned image buffers  (64-byte alignment, matches aligned_alloc advice in hints)
// ---------------------------------------------------------------------------
static uint8_t*  alloc_u8 (int n) { return static_cast<uint8_t* >(aligned_alloc(64, n)); }
static int16_t*  alloc_i16(int n) { return static_cast<int16_t* >(aligned_alloc(64, n * sizeof(int16_t))); }
static uint8_t*  alloc_mag(int n) { return static_cast<uint8_t* >(aligned_alloc(64, n)); }

// ===========================================================================
// SECTION A — Sobel gradient tests (Phase 2.3)
// ===========================================================================

// A1. Uniform image → zero gradient everywhere
TEST(SobelTest, UniformImageZeroGradient) {
    const int W = 32, H = 32;
    uint8_t*  img = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    // Fill with constant value 128
    memset(img, 128, W * H);
    compute_sobel(img, Gx, Gy, W, H);

    // Interior pixels must be exactly zero (boundary pixels may differ due to zero-padding)
    for (int r = 1; r < H - 1; ++r) {
        for (int c = 1; c < W - 1; ++c) {
            EXPECT_EQ(Gx[r * W + c], 0) << "Gx non-zero at (" << r << "," << c << ")";
            EXPECT_EQ(Gy[r * W + c], 0) << "Gy non-zero at (" << r << "," << c << ")";
        }
    }

    free(img); free(Gx); free(Gy);
}

// A2. Sharp vertical edge (left half = 0, right half = 255)
//     → interior pixels ON the boundary should have large |Gx| and near-zero |Gy|
TEST(SobelTest, VerticalEdgeLargeGxSmallGy) {
    const int W = 32, H = 32;
    uint8_t*  img = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    // Create a vertical edge at column W/2
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            img[r * W + c] = (c < W / 2) ? 0 : 255;
        }
    }

    compute_sobel(img, Gx, Gy, W, H);

    // At the edge column (W/2), interior rows should have large positive Gx
    // and very small Gy (the edge is perfectly horizontal in the image sense)
    int edge_col = W / 2;
    for (int r = 2; r < H - 2; ++r) {  // skip 2-px border to avoid zero-pad artefacts
        int16_t gx_val = Gx[r * W + edge_col];
        int16_t gy_val = Gy[r * W + edge_col];
        EXPECT_GT(gx_val, 0)   << "Expected positive Gx at edge, row " << r;
        EXPECT_EQ(gy_val, 0)   << "Expected zero Gy at vertical edge, row " << r;
    }

    // Far from the edge, both gradients should be zero
    for (int r = 2; r < H - 2; ++r) {
        EXPECT_EQ(Gx[r * W + 2],     0) << "Gx non-zero in flat region, row " << r;
        EXPECT_EQ(Gy[r * W + 2],     0) << "Gy non-zero in flat region, row " << r;
        EXPECT_EQ(Gx[r * W + W - 3], 0) << "Gx non-zero in flat region right, row " << r;
    }

    free(img); free(Gx); free(Gy);
}

// A3. Sharp horizontal edge (top half = 0, bottom half = 255)
//     → large |Gy| at the boundary, near-zero |Gx|
TEST(SobelTest, HorizontalEdgeLargeGySmallGx) {
    const int W = 32, H = 32;
    uint8_t*  img = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    // Horizontal edge at row H/2
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            img[r * W + c] = (r < H / 2) ? 0 : 255;
        }
    }

    compute_sobel(img, Gx, Gy, W, H);

    int edge_row = H / 2;
    for (int c = 2; c < W - 2; ++c) {
        int16_t gx_val = Gx[edge_row * W + c];
        int16_t gy_val = Gy[edge_row * W + c];
        EXPECT_GT(gy_val, 0) << "Expected positive Gy at horizontal edge, col " << c;
        EXPECT_EQ(gx_val, 0) << "Expected zero Gx at horizontal edge, col " << c;
    }

    free(img); free(Gx); free(Gy);
}

// A4. Diagonal edge (upper-left quadrant = 0, lower-right = 255, step along diagonal)
//     → both Gx and Gy should be significant along the diagonal
TEST(SobelTest, DiagonalEdgeBothGradientsSignificant) {
    const int W = 64, H = 64;
    uint8_t*  img = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    // Simple diagonal: pixel = 0 if r > c, else 255
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            img[r * W + c] = (r > c) ? 0 : 255;
        }
    }

    compute_sobel(img, Gx, Gy, W, H);

    // Sample a few pixels directly ON the diagonal (interior only)
    int non_zero_count = 0;
    for (int d = 4; d < W - 4; ++d) {
        int16_t gx = Gx[d * W + d];
        int16_t gy = Gy[d * W + d];
        if (std::abs(gx) > 0 || std::abs(gy) > 0) ++non_zero_count;
    }
    EXPECT_GT(non_zero_count, 10) << "Diagonal edge should produce non-zero gradients";

    free(img); free(Gx); free(Gy);
}

// A5. All-black image → all gradients should be zero
TEST(SobelTest, AllBlackImageZeroGradient) {
    const int W = 16, H = 16;
    uint8_t*  img = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);

    memset(img, 0, W * H);
    compute_sobel(img, Gx, Gy, W, H);

    for (int i = 0; i < W * H; ++i) {
        EXPECT_EQ(Gx[i], 0);
        EXPECT_EQ(Gy[i], 0);
    }

    free(img); free(Gx); free(Gy);
}

// ===========================================================================
// SECTION B — Gradient magnitude tests (Phase 2.4)
// ===========================================================================

// B1. Zero gradients → magnitude should be all zero (L1 and L2)
TEST(MagnitudeTest, ZeroGradientZeroMagnitude_L1) {
    const int W = 16, H = 16;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_mag(W * H);

    memset(Gx, 0, W * H * sizeof(int16_t));
    memset(Gy, 0, W * H * sizeof(int16_t));
    gradient_magnitude_l1(Gx, Gy, mag, W, H);

    for (int i = 0; i < W * H; ++i) {
        EXPECT_EQ(mag[i], 0);
    }

    free(Gx); free(Gy); free(mag);
}

TEST(MagnitudeTest, ZeroGradientZeroMagnitude_L2) {
    const int W = 16, H = 16;
    int16_t* Gx  = alloc_i16(W * H);
    int16_t* Gy  = alloc_i16(W * H);
    uint8_t* mag = alloc_mag(W * H);

    memset(Gx, 0, W * H * sizeof(int16_t));
    memset(Gy, 0, W * H * sizeof(int16_t));
    gradient_magnitude_l2(Gx, Gy, mag, W, H);

    for (int i = 0; i < W * H; ++i) {
        EXPECT_EQ(mag[i], 0);
    }

    free(Gx); free(Gy); free(mag);
}

// B2. Non-zero input → magnitude should NOT be all zero
TEST(MagnitudeTest, NonZeroGradientNonZeroMagnitude_L1) {
    const int W = 32, H = 32;
    uint8_t*  img = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);
    uint8_t*  mag = alloc_mag(W * H);

    // Vertical edge → non-zero Gx
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r * W + c] = (c < W / 2) ? 0 : 255;

    compute_sobel(img, Gx, Gy, W, H);
    gradient_magnitude_l1(Gx, Gy, mag, W, H);

    int non_zero = 0;
    for (int i = 0; i < W * H; ++i) if (mag[i] > 0) ++non_zero;
    EXPECT_GT(non_zero, 0) << "L1 magnitude should have non-zero pixels on an edged image";

    free(img); free(Gx); free(Gy); free(mag);
}

TEST(MagnitudeTest, NonZeroGradientNonZeroMagnitude_L2) {
    const int W = 32, H = 32;
    uint8_t*  img = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);
    uint8_t*  mag = alloc_mag(W * H);

    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r * W + c] = (c < W / 2) ? 0 : 255;

    compute_sobel(img, Gx, Gy, W, H);
    gradient_magnitude_l2(Gx, Gy, mag, W, H);

    int non_zero = 0;
    for (int i = 0; i < W * H; ++i) if (mag[i] > 0) ++non_zero;
    EXPECT_GT(non_zero, 0) << "L2 magnitude should have non-zero pixels on an edged image";

    free(img); free(Gx); free(Gy); free(mag);
}

// B3. Normalization: output must always be in [0, 255]
TEST(MagnitudeTest, OutputInRange_L1) {
    const int W = 48, H = 48;  // non-power-of-two size to catch strip-mining edge cases
    uint8_t*  img = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);
    uint8_t*  mag = alloc_mag(W * H);

    // Checkerboard pattern — lots of gradients
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r * W + c] = ((r + c) % 2 == 0) ? 0 : 255;

    compute_sobel(img, Gx, Gy, W, H);
    gradient_magnitude_l1(Gx, Gy, mag, W, H);

    for (int i = 0; i < W * H; ++i) {
        EXPECT_GE(mag[i], 0)   << "Magnitude below 0 at index " << i;
        EXPECT_LE(mag[i], 255) << "Magnitude above 255 at index " << i;
    }

    free(img); free(Gx); free(Gy); free(mag);
}

TEST(MagnitudeTest, OutputInRange_L2) {
    const int W = 48, H = 48;
    uint8_t*  img = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);
    uint8_t*  mag = alloc_mag(W * H);

    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r * W + c] = ((r + c) % 2 == 0) ? 0 : 255;

    compute_sobel(img, Gx, Gy, W, H);
    gradient_magnitude_l2(Gx, Gy, mag, W, H);

    for (int i = 0; i < W * H; ++i) {
        EXPECT_GE(mag[i], 0)   << "Magnitude below 0 at index " << i;
        EXPECT_LE(mag[i], 255) << "Magnitude above 255 at index " << i;
    }

    free(img); free(Gx); free(Gy); free(mag);
}

// B4. L1 ≥ L2 property: for a 45° edge L1 = sqrt(2)*L2, otherwise L1 ≤ sqrt(2)*L2
//     At minimum we verify L1 >= L2 before normalisation on a controlled input
TEST(MagnitudeTest, L1AlwaysGreaterOrEqualToL2_RawCheck) {
    // Directly check that |Gx| + |Gy| >= sqrt(Gx^2 + Gy^2)  (always true by triangle inequality)
    // We construct Gx/Gy values manually and compare un-normalised magnitudes
    const int N = 16;
    int16_t Gx_vals[N] = {10, 0, -10, 5, 100, 0, 0, 7, 50, 30, 200, 100, 1, 0, -50, 255};
    int16_t Gy_vals[N] = {0, 10,  0, 5,   0, 100, 50, 7, 30, 50,  50, 100, 0, 0,  50,   0};

    for (int i = 0; i < N; ++i) {
        double l1 = std::abs((int)Gx_vals[i]) + std::abs((int)Gy_vals[i]);
        double l2 = std::sqrt((double)Gx_vals[i]*Gx_vals[i] + (double)Gy_vals[i]*Gy_vals[i]);
        EXPECT_GE(l1, l2 - 1e-9) << "L1 < L2 at index " << i
            << " Gx=" << Gx_vals[i] << " Gy=" << Gy_vals[i];
    }
}

// ===========================================================================
// SECTION C — Gradient direction tests (Phase 2.5)
// ===========================================================================

// C1. Vertical edge (large Gx, Gy=0) → direction should be DIR_0
TEST(DirectionTest, VerticalEdgeIsDir0) {
    const int W = 32, H = 32;
    uint8_t*  img = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);
    uint8_t*  dir = alloc_mag(W * H);

    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r * W + c] = (c < W / 2) ? 0 : 255;

    compute_sobel(img, Gx, Gy, W, H);
    gradient_direction(Gx, Gy, dir, W, H);

    // Interior pixels on the edge column should be DIR_0 (horizontal gradient)
    int edge_col = W / 2;
    for (int r = 2; r < H - 2; ++r) {
        if (Gx[r * W + edge_col] != 0 || Gy[r * W + edge_col] != 0) {
            EXPECT_EQ(dir[r * W + edge_col], DIR_0)
                << "Vertical edge pixel at (" << r << "," << edge_col << ") should be DIR_0";
        }
    }

    free(img); free(Gx); free(Gy); free(dir);
}

// C2. Horizontal edge (Gx=0, large Gy) → direction should be DIR_90
TEST(DirectionTest, HorizontalEdgeIsDir90) {
    const int W = 32, H = 32;
    uint8_t*  img = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);
    uint8_t*  dir = alloc_mag(W * H);

    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r * W + c] = (r < H / 2) ? 0 : 255;

    compute_sobel(img, Gx, Gy, W, H);
    gradient_direction(Gx, Gy, dir, W, H);

    int edge_row = H / 2;
    for (int c = 2; c < W - 2; ++c) {
        if (Gx[edge_row * W + c] != 0 || Gy[edge_row * W + c] != 0) {
            EXPECT_EQ(dir[edge_row * W + c], DIR_90)
                << "Horizontal edge pixel at (" << edge_row << "," << c << ") should be DIR_90";
        }
    }

    free(img); free(Gx); free(Gy); free(dir);
}

// C3. Controlled Gx/Gy values → verify direction logic directly
TEST(DirectionTest, ControlledValues) {
    // Test the boundary conditions of the direction logic directly
    // Pixel 0: Gx=100, Gy=0    → nearly horizontal → DIR_0
    // Pixel 1: Gx=0,   Gy=100  → nearly vertical   → DIR_90
    // Pixel 2: Gx=50,  Gy=50   → 45 degrees        → DIR_45 or DIR_135
    // Pixel 3: Gx=50,  Gy=-50  → 135 degrees       → DIR_45 or DIR_135

    const int N = 4;
    int16_t Gx_vals[N] = {100, 0,  50,  50};
    int16_t Gy_vals[N] = {  0, 100, 50, -50};
    uint8_t dir[N];

    gradient_direction(Gx_vals, Gy_vals, dir, N, 1);

    EXPECT_EQ(dir[0], DIR_0)  << "Pure horizontal gradient should be DIR_0";
    EXPECT_EQ(dir[1], DIR_90) << "Pure vertical gradient should be DIR_90";
    // Pixels 2 and 3 are diagonal (45 or 135) — check they are NOT 0 or 90
    EXPECT_NE(dir[2], DIR_0)  << "Diagonal pixel 2 should not be DIR_0";
    EXPECT_NE(dir[2], DIR_90) << "Diagonal pixel 2 should not be DIR_90";
    EXPECT_NE(dir[3], DIR_0)  << "Diagonal pixel 3 should not be DIR_0";
    EXPECT_NE(dir[3], DIR_90) << "Diagonal pixel 3 should not be DIR_90";

    // Pixel 2: Gx>0, Gy>0 → DIR_135 (same sign)
    EXPECT_EQ(dir[2], DIR_135);
    // Pixel 3: Gx>0, Gy<0 → DIR_45 (opposite sign)
    EXPECT_EQ(dir[3], DIR_45);
}

// C4. All-zero gradient → direction can be anything (no crash, valid output)
TEST(DirectionTest, ZeroGradientNoCrash) {
    const int N = 16;
    int16_t Gx[N] = {}, Gy[N] = {};
    uint8_t dir[N];

    EXPECT_NO_FATAL_FAILURE(gradient_direction(Gx, Gy, dir, N, 1));

    for (int i = 0; i < N; ++i) {
        EXPECT_LE(dir[i], 3) << "Direction out of range at index " << i;
    }
}

// C5. All output values are in {0, 1, 2, 3}  (random-ish checkerboard input)
TEST(DirectionTest, AllOutputsValidRange) {
    const int W = 48, H = 48;
    uint8_t*  img = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);
    uint8_t*  dir = alloc_mag(W * H);

    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r * W + c] = (uint8_t)((r * 7 + c * 13) % 256);

    compute_sobel(img, Gx, Gy, W, H);
    gradient_direction(Gx, Gy, dir, W, H);

    for (int i = 0; i < W * H; ++i) {
        EXPECT_LE(dir[i], 3) << "Direction value out of [0..3] at index " << i;
    }

    free(img); free(Gx); free(Gy); free(dir);
}

// ===========================================================================
// SECTION D — Integration: full pipeline sanity check
// ===========================================================================

// D1. End-to-end on a vertical edge: verify the complete chain produces
//     non-zero magnitude at the edge and zero in the flat interior.
//
// Note on normalisation: gradient_magnitude_l2 normalises so that the global
// maximum in the image becomes 255.  For a vertical step-edge, the highest
// Sobel response can appear at the border pixels (due to zero-padding bringing
// a partial sum) rather than exactly at the interior edge column.  So the
// interior edge column pixels are "large" but not necessarily 255.
// The important invariant is:  edge_col >> flat_interior.
TEST(IntegrationTest, VerticalEdgePipelineEndToEnd) {
    const int W = 32, H = 32;
    uint8_t*  img = alloc_u8(W * H);
    int16_t*  Gx  = alloc_i16(W * H);
    int16_t*  Gy  = alloc_i16(W * H);
    uint8_t*  mag = alloc_mag(W * H);
    uint8_t*  dir = alloc_mag(W * H);

    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r * W + c] = (c < W / 2) ? 0 : 255;

    compute_sobel(img, Gx, Gy, W, H);
    gradient_magnitude_l2(Gx, Gy, mag, W, H);
    gradient_direction(Gx, Gy, dir, W, H);

    // Interior edge column: magnitude should be large (> 200 after normalisation)
    int edge_col = W / 2;
    for (int r = 2; r < H - 2; ++r) {
        EXPECT_GT(mag[r * W + edge_col], 200)
            << "Normalised magnitude should be large at edge column, row " << r;
    }

    // Flat interior (well away from edge and border): magnitude should be 0
    // Use columns 4 and W-5 to be safely away from the zero-pad border effect
    for (int r = 2; r < H - 2; ++r) {
        EXPECT_EQ(mag[r * W + 4],     0)
            << "Magnitude should be 0 in flat left region, row " << r;
        EXPECT_EQ(mag[r * W + W - 5], 0)
            << "Magnitude should be 0 in flat right region, row " << r;
    }

    free(img); free(Gx); free(Gy); free(mag); free(dir);
}
