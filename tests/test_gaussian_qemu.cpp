// =============================================================================
// test_gaussian_qemu.cpp
// Phase 3.2 — QEMU-side equivalence tests with MS Counting (Emulation Time)
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <chrono> // Required for measuring milliseconds
#include "canny.h" 

#define QEMU_EXPECT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("FAIL: %s\n", msg); \
            exit(1); \
        } else { \
            printf("PASS: %s\n", msg); \
        } \
    } while(0)

// Test 1: Uniform Image (اختبار الصورة الموحدة الإضاءة)
void test_uniform_image() {
    const int width = 16;
    const int height = 16;
    uint8_t* input = (uint8_t*)malloc(width * height);
    uint8_t* output = (uint8_t*)malloc(width * height);

    for (int i = 0; i < width * height; ++i) {
        input[i] = 128;
        output[i] = 0;
    }

    auto start_wall = std::chrono::high_resolution_clock::now();
    gaussian_blur_separable<uint8_t, int32_t, int16_t>(input, output, width, height);
    auto end_wall = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> wall_ms = end_wall - start_wall;

    bool passed = true;
    for (int y = 2; y < height - 2; ++y) {
        for (int x = 2; x < width - 2; ++x) {
            int diff = output[y * width + x] - 128;
            if (diff < -1 || diff > 1) passed = false;
        }
    }
    
    QEMU_EXPECT_TRUE(passed, "1. Uniform Image Test");
    printf("  -> Emulation Time: %f ms\n\n", wall_ms.count());

    free(input);
    free(output);
}

// Test 2: All Black Image (اختبار الصورة السوداء بالكامل)
void test_black_image() {
    const int width = 16;
    const int height = 16;
    uint8_t* input = (uint8_t*)malloc(width * height);
    uint8_t* output = (uint8_t*)malloc(width * height);

    for (int i = 0; i < width * height; ++i) {
        input[i] = 0;
        output[i] = 255;
    }

    auto start_wall = std::chrono::high_resolution_clock::now();
    gaussian_blur_separable<uint8_t, int32_t, int16_t>(input, output, width, height);
    auto end_wall = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> wall_ms = end_wall - start_wall;

    bool passed = true;
    for (int i = 0; i < width * height; ++i) {
        if (output[i] != 0) passed = false;
    }
    
    QEMU_EXPECT_TRUE(passed, "2. All Black Image Test");
    printf("  -> Emulation Time: %f ms\n\n", wall_ms.count());

    free(input);
    free(output);
}

// Test 3: Impulse Symmetry (اختبار النبضة وتماثل انتشار التمويه)
void test_impulse_symmetry() {
    const int width = 15;
    const int height = 15;
    const int cx = 7;
    const int cy = 7;

    uint8_t* input = (uint8_t*)malloc(width * height);
    uint8_t* output = (uint8_t*)malloc(width * height);

    for (int i = 0; i < width * height; ++i) {
        input[i] = 0;
        output[i] = 0;
    }
    input[cy * width + cx] = 255;

    auto start_wall = std::chrono::high_resolution_clock::now();
    gaussian_blur_separable<uint8_t, int32_t, int16_t>(input, output, width, height);
    auto end_wall = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> wall_ms = end_wall - start_wall;

    bool passed = true;
    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            uint8_t val1 = output[(cy + dy) * width + (cx + dx)];
            uint8_t val_mirror_x = output[(cy + dy) * width + (cx - dx)];
            uint8_t val_mirror_y = output[(cy - dy) * width + (cx + dx)];
            uint8_t val_diagonal = output[(cy + dx) * width + (cx + dy)];

            if (val1 != val_mirror_x || val1 != val_mirror_y || val1 != val_diagonal) {
                passed = false;
            }
        }
    }

    QEMU_EXPECT_TRUE(passed, "3. Impulse Symmetry Test");
    printf("  -> Emulation Time: %f ms\n\n", wall_ms.count());

    free(input);
    free(output);
}

int main() {
    printf("--- Running QEMU Gaussian Blur Sanity Tests ---\n\n");
    test_uniform_image();
    test_black_image();
    test_impulse_symmetry();
    printf("ALL TESTS PASSED\n");
    return 0;
}