// =============================================================================
// test_gaussian_qemu_rvv.cpp
// Phase 6.6 — QEMU-side Gaussian Blur Standalone Property Tests & Timing
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <chrono>

// Forward declaration of the RVV template implementation from phase_6_4.cpp
template <class PixelT, class AccumulatorT, class KernelT>
void gaussian_blur_2d(const PixelT* __restrict input, PixelT* __restrict output, size_t width, size_t height);

// ---------------------------------------------------------------------------
// Test tracking helpers
// ---------------------------------------------------------------------------
static int tests_run    = 0;
static int tests_passed = 0;

#define QEMU_EXPECT_TRUE(cond, msg) do {                            \
    ++tests_run;                                                     \
    if (!(cond)) {                                                   \
        printf("  FAIL [%s]: condition was false\n", (msg));          \
    } else {                                                         \
        ++tests_passed;                                              \
    }                                                                \
} while(0)

// Allocate 64-byte aligned buffers (required for RVV load intrinsics)
static uint8_t* au8(int n) { return (uint8_t*)aligned_alloc(64, n); }

// ===========================================================================
// Test Cases
// ===========================================================================

// 1. All Black Image Test
static void test_all_black_image() {
    printf("  T1: All Black Image Test (100x75 non-pow2)...\n");
    const int W = 100;
    const int H = 75;
    const int total = W * H;

    uint8_t* img = au8(total);
    uint8_t* out = au8(total);

    std::memset(img, 0, total);
    std::memset(out, 0xFF, total); // Fill with junk to ensure overwrite

    // Time the execution
    auto start = std::chrono::high_resolution_clock::now();
    gaussian_blur_2d<uint8_t, int32_t, int16_t>(img, out, W, H);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    // Verify all pixels remain 0
    bool passed = true;
    for (int i = 0; i < total; ++i) {
        if (out[i] != 0) {
            passed = false;
            printf("      Mismatch at index %d: expected 0, got %d\n", i, out[i]);
            break;
        }
    }

    printf("      Execution Time: %.2f us\n", elapsed.count());
    QEMU_EXPECT_TRUE(passed, "T1: All Black Image produces all zeros");

    free(img); free(out);
}

// 2. Uniform Image Test
static void test_uniform_image() {
    printf("  T2: Uniform Image Test (100x75 non-pow2)...\n");
    const int W = 100;
    const int H = 75;
    const int total = W * H;
    const uint8_t fill_val = 128;

    uint8_t* img = au8(total);
    uint8_t* out = au8(total);

    std::memset(img, fill_val, total);
    std::memset(out, 0, total);

    // Time the execution
    auto start = std::chrono::high_resolution_clock::now();
    gaussian_blur_2d<uint8_t, int32_t, int16_t>(img, out, W, H);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    // Verify interior pixels (away from the zero-padded borders) equal the fill value
    bool passed = true;
    int err_count = 0;
    for (int r = 2; r < H - 2; ++r) {
        for (int c = 2; c < W - 2; ++c) {
            int idx = r * W + c;
            // Allow +/- 1 tolerance due to integer division truncation vs rounding differences
            if (std::abs((int)out[idx] - (int)fill_val) > 1) {
                passed = false;
                if (err_count < 5) {
                    printf("      Mismatch at row %d, col %d: expected ~%d, got %d\n", r, c, fill_val, out[idx]);
                }
                err_count++;
            }
        }
    }

    printf("      Execution Time: %.2f us\n", elapsed.count());
    QEMU_EXPECT_TRUE(passed, "T2: Uniform image interior matches input within +/- 1");

    free(img); free(out);
}

// 3. Impulse Symmetry Test
static void test_impulse_symmetry() {
    printf("  T3: Impulse Symmetry Test (100x75 non-pow2)...\n");
    const int W = 100;
    const int H = 75;
    const int total = W * H;

    uint8_t* img = au8(total);
    uint8_t* out = au8(total);

    std::memset(img, 0, total);
    std::memset(out, 0, total);

    // Place an impulse pixel right in the center
    int cx = W / 2;
    int cy = H / 2;
    img[cy * W + cx] = 255;

    // Time the execution
    auto start = std::chrono::high_resolution_clock::now();
    gaussian_blur_2d<uint8_t, int32_t, int16_t>(img, out, W, H);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;

    // Verify 2D Quadrant Symmetry around the impulse center (within the 5x5 neighborhood window)
    bool passed = true;
    for (int dy = 0; dy <= 2; ++dy) {
        for (int dx = 0; dx <= 2; ++dx) {
            uint8_t q1 = out[(cy - dy) * W + (cx + dx)]; // Top-Right
            uint8_t q2 = out[(cy - dy) * W + (cx - dx)]; // Top-Left
            uint8_t q3 = out[(cy + dy) * W + (cx - dx)]; // Bottom-Left
            uint8_t q4 = out[(cy + dy) * W + (cx + dx)]; // Bottom-Right

            if (q1 != q2 || q1 != q3 || q1 != q4) {
                passed = false;
                printf("      Asymmetry detected at offset (+/-%d, +/-%d): Q1=%d, Q2=%d, Q3=%d, Q4=%d\n", 
                       dx, dy, q1, q2, q3, q4);
                break;
            }
        }
        if (!passed) break;
    }

    printf("      Execution Time: %.2f us\n", elapsed.count());
    QEMU_EXPECT_TRUE(passed, "T3: Blur response around center impulse is perfectly symmetric");

    free(img); free(out);
}

// ===========================================================================
// Main Entry Point
// ===========================================================================
int main() {
    printf("=== QEMU-side Standalone RVV Gaussian Property Tests ===\n");

    test_all_black_image();
    test_uniform_image();
    test_impulse_symmetry();

    printf("\nResults: %d / %d passed\n", tests_passed, tests_run);

    if (tests_passed == tests_run) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
}