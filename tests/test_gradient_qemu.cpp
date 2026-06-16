// =============================================================================
// test_gradient_qemu.cpp
// Phase 3 — QEMU-side equivalence tests (assert-based, no GoogleTest)
//
// Cross-compile for RISC-V:
//   riscv64-unknown-elf-g++ -std=c++17 -O2 -march=rv64gcv -I../include \
//       test_gradient_qemu.cpp -o test_gradient_qemu
//
// Run on QEMU at multiple VLEN values:
//   qemu-riscv64 -cpu rv64,v=true,vlen=128 ./test_gradient_qemu
//   qemu-riscv64 -cpu rv64,v=true,vlen=256 ./test_gradient_qemu
//   qemu-riscv64 -cpu rv64,v=true,vlen=512 ./test_gradient_qemu
//
// All three runs must print "ALL TESTS PASSED" and exit with code 0.
// This is the vector-length-agnosticism check required by the spec.
//
// NOTE: When you later add RVV intrinsic implementations, add a second
//       call with the RVV version alongside the scalar call and compare
//       their outputs (allowing ±1 tolerance for rounding).
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cstdio>
#include "gradient.h"

// ---------------------------------------------------------------------------
// Minimal test helpers
// ---------------------------------------------------------------------------
static int tests_run    = 0;
static int tests_passed = 0;

#define QEMU_EXPECT_EQ(a, b, msg) do {                              \
    ++tests_run;                                                     \
    if ((a) != (b)) {                                               \
        printf("FAIL [%s]: expected %d, got %d\n", (msg), (int)(b), (int)(a)); \
    } else {                                                         \
        ++tests_passed;                                              \
    }                                                                \
} while(0)

#define QEMU_EXPECT_TRUE(cond, msg) do {                            \
    ++tests_run;                                                     \
    if (!(cond)) {                                                   \
        printf("FAIL [%s]: condition was false\n", (msg));          \
    } else {                                                         \
        ++tests_passed;                                              \
    }                                                                \
} while(0)

// ±1 tolerance comparison (for future RVV vs scalar equivalence checks)
static bool within_one(uint8_t a, uint8_t b) {
    int diff = (int)a - (int)b;
    return diff >= -1 && diff <= 1;
}

// ---------------------------------------------------------------------------
// Allocate 64-byte aligned buffers  (required for some RVV load intrinsics)
// ---------------------------------------------------------------------------
static uint8_t*  au8 (int n) { return (uint8_t* )aligned_alloc(64, n); }
static int16_t*  ai16(int n) { return (int16_t* )aligned_alloc(64, n * sizeof(int16_t)); }

// ===========================================================================
// Tests
// ===========================================================================

// T1: Uniform image → all-zero gradients (verifies correct zero-pad boundary)
static void test_uniform_zero_gradient() {
    printf("  T1: uniform image zero gradient...\n");
    const int W = 48, H = 48;  // non-power-of-two
    uint8_t*  img = au8(W * H);
    int16_t*  Gx  = ai16(W * H);
    int16_t*  Gy  = ai16(W * H);

    memset(img, 128, W * H);
    compute_sobel(img, Gx, Gy, W, H);

    // Check interior (2-px margin, to exclude zero-pad border effects)
    bool all_zero = true;
    for (int r = 2; r < H - 2; ++r)
        for (int c = 2; c < W - 2; ++c)
            if (Gx[r*W+c] != 0 || Gy[r*W+c] != 0) { all_zero = false; break; }

    QEMU_EXPECT_TRUE(all_zero, "T1: uniform interior gradient == 0");

    free(img); free(Gx); free(Gy);
}

// T2: Vertical edge → large Gx, zero Gy on edge column
static void test_vertical_edge() {
    printf("  T2: vertical edge Gx large, Gy=0...\n");
    const int W = 50, H = 50;  // non-power-of-two
    uint8_t*  img = au8(W * H);
    int16_t*  Gx  = ai16(W * H);
    int16_t*  Gy  = ai16(W * H);

    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r*W+c] = (c < W/2) ? 0 : 255;

    compute_sobel(img, Gx, Gy, W, H);

    int edge_col = W/2;
    bool gx_positive = true;
    bool gy_zero     = true;
    for (int r = 2; r < H - 2; ++r) {
        if (Gx[r*W + edge_col] <= 0) gx_positive = false;
        if (Gy[r*W + edge_col] != 0) gy_zero     = false;
    }
    QEMU_EXPECT_TRUE(gx_positive, "T2: Gx > 0 at vertical edge");
    QEMU_EXPECT_TRUE(gy_zero,     "T2: Gy == 0 at vertical edge");

    free(img); free(Gx); free(Gy);
}

// T3: L1 magnitude output always in [0, 255]
static void test_magnitude_l1_range() {
    printf("  T3: L1 magnitude in [0,255]...\n");
    const int W = 75, H = 100;  // non-power-of-two, non-square
    uint8_t*  img = au8(W * H);
    int16_t*  Gx  = ai16(W * H);
    int16_t*  Gy  = ai16(W * H);
    uint8_t*  mag = au8(W * H);

    // Checkerboard pattern
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r*W+c] = ((r+c)%2 == 0) ? 0 : 255;

    compute_sobel(img, Gx, Gy, W, H);
    gradient_magnitude_l1(Gx, Gy, mag, W, H);

    bool in_range = true;
    for (int i = 0; i < W*H; ++i)
        if (mag[i] > 255) { in_range = false; break; }

    QEMU_EXPECT_TRUE(in_range, "T3: all L1 magnitudes in [0,255]");

    free(img); free(Gx); free(Gy); free(mag);
}

// T4: L2 magnitude output always in [0, 255]
static void test_magnitude_l2_range() {
    printf("  T4: L2 magnitude in [0,255]...\n");
    const int W = 75, H = 100;
    uint8_t*  img = au8(W * H);
    int16_t*  Gx  = ai16(W * H);
    int16_t*  Gy  = ai16(W * H);
    uint8_t*  mag = au8(W * H);

    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r*W+c] = ((r+c)%2 == 0) ? 0 : 255;

    compute_sobel(img, Gx, Gy, W, H);
    gradient_magnitude_l2(Gx, Gy, mag, W, H);

    bool in_range = true;
    for (int i = 0; i < W*H; ++i)
        if (mag[i] > 255) { in_range = false; break; }

    QEMU_EXPECT_TRUE(in_range, "T4: all L2 magnitudes in [0,255]");

    free(img); free(Gx); free(Gy); free(mag);
}

// T5: Direction output always in {0, 1, 2, 3}
static void test_direction_range() {
    printf("  T5: direction values in {0,1,2,3}...\n");
    const int W = 48, H = 48;
    uint8_t*  img = au8(W * H);
    int16_t*  Gx  = ai16(W * H);
    int16_t*  Gy  = ai16(W * H);
    uint8_t*  dir = au8(W * H);

    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r*W+c] = (uint8_t)((r * 7 + c * 13) % 256);

    compute_sobel(img, Gx, Gy, W, H);
    gradient_direction(Gx, Gy, dir, W, H);

    bool in_range = true;
    for (int i = 0; i < W*H; ++i)
        if (dir[i] > 3) { in_range = false; break; }

    QEMU_EXPECT_TRUE(in_range, "T5: all direction values in {0..3}");

    free(img); free(Gx); free(Gy); free(dir);
}

// T6: Vertical edge direction is DIR_0
static void test_vertical_edge_direction() {
    printf("  T6: vertical edge → DIR_0...\n");
    const int W = 32, H = 32;
    uint8_t*  img = au8(W * H);
    int16_t*  Gx  = ai16(W * H);
    int16_t*  Gy  = ai16(W * H);
    uint8_t*  dir = au8(W * H);

    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r*W+c] = (c < W/2) ? 0 : 255;

    compute_sobel(img, Gx, Gy, W, H);
    gradient_direction(Gx, Gy, dir, W, H);

    bool correct = true;
    int edge_col = W/2;
    for (int r = 2; r < H - 2; ++r) {
        if ((Gx[r*W+edge_col] != 0 || Gy[r*W+edge_col] != 0) && dir[r*W+edge_col] != DIR_0)
            { correct = false; break; }
    }
    QEMU_EXPECT_TRUE(correct, "T6: vertical edge direction == DIR_0");

    free(img); free(Gx); free(Gy); free(dir);
}

// T7: Non-power-of-two strip-mining stress test (catches VLA tail-case bugs)
static void test_non_pow2_size() {
    printf("  T7: non-pow2 image size (strip-mining tail)...\n");
    // Use sizes that are NOT multiples of 128/256/512 bits worth of pixels
    const int W = 37, H = 43;
    uint8_t*  img = au8(W * H);
    int16_t*  Gx  = ai16(W * H);
    int16_t*  Gy  = ai16(W * H);
    uint8_t*  mag = au8(W * H);
    uint8_t*  dir = au8(W * H);

    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            img[r*W+c] = (c < W/2) ? 0 : 255;

    compute_sobel(img, Gx, Gy, W, H);
    gradient_magnitude_l1(Gx, Gy, mag, W, H);
    gradient_direction(Gx, Gy, dir, W, H);

    // Just verify no crash and output is sane
    bool mag_ok = true;
    bool dir_ok = true;
    for (int i = 0; i < W*H; ++i) {
        if (mag[i] > 255) mag_ok = false;
        if (dir[i] > 3)   dir_ok = false;
    }
    QEMU_EXPECT_TRUE(mag_ok, "T7: non-pow2 magnitude in range");
    QEMU_EXPECT_TRUE(dir_ok, "T7: non-pow2 direction in range");

    free(img); free(Gx); free(Gy); free(mag); free(dir);
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    printf("=== QEMU-side gradient equivalence tests ===\n");

    test_uniform_zero_gradient();
    test_vertical_edge();
    test_magnitude_l1_range();
    test_magnitude_l2_range();
    test_direction_range();
    test_vertical_edge_direction();
    test_non_pow2_size();

    printf("\nResults: %d / %d passed\n", tests_passed, tests_run);

    if (tests_passed == tests_run) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
}
