#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "gradient.h"
#include "gradient_rvv.h"

static inline uint64_t read_cycles() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r"(cycles));
    return cycles;
}

static void generate_test_image(uint8_t* img, int width, int height) {
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            // Non-trivial synthetic image with edges and changing pattern.
            if (c < width / 2)
                img[r * width + c] = (uint8_t)((r + c) % 128);
            else
                img[r * width + c] = (uint8_t)(128 + ((r + c) % 128));
        }
    }
}

static uint64_t checksum_u8(const uint8_t* data, int n) {
    uint64_t s = 0;
    for (int i = 0; i < n; ++i) {
        s += data[i];
    }
    return s;
}

int main() {
    // Non-power-of-two size to test RVV strip-mining tail behavior.
    const int width = 100;
    const int height = 75;
    const int n = width * height;
    const int iterations = 200;

    uint8_t* input = (uint8_t*)malloc(n);
    int16_t* gx = (int16_t*)malloc(n * sizeof(int16_t));
    int16_t* gy = (int16_t*)malloc(n * sizeof(int16_t));
    uint8_t* scalar_mag = (uint8_t*)malloc(n);
    uint8_t* rvv_mag = (uint8_t*)malloc(n);

    if (!input || !gx || !gy || !scalar_mag || !rvv_mag) {
        printf("Allocation failed\n");
        return 1;
    }

    generate_test_image(input, width, height);

    // Compute Sobel gradients using existing scalar function.
    compute_sobel(input, gx, gy, width, height);

    // Reference scalar L1 magnitude.
    gradient_magnitude_l1(gx, gy, scalar_mag, width, height);

    // New RVV L1 magnitude.
    gradient_magnitude_l1_rvv(gx, gy, rvv_mag, width, height);

    // Compare scalar vs RVV output.
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (scalar_mag[i] != rvv_mag[i]) {
            if (errors < 10) {
                printf("Mismatch at index %d: scalar=%u rvv=%u\n",
                       i,
                       (unsigned)scalar_mag[i],
                       (unsigned)rvv_mag[i]);
            }
            errors++;
        }
    }

    uint64_t scalar_checksum = checksum_u8(scalar_mag, n);
    uint64_t rvv_checksum = checksum_u8(rvv_mag, n);

    printf("Phase 6.5 RVV L1 Magnitude Test\n");
    printf("Image size: %d x %d\n", width, height);
    printf("Scalar checksum: %llu\n", (unsigned long long)scalar_checksum);
    printf("RVV checksum: %llu\n", (unsigned long long)rvv_checksum);

    if (errors == 0) {
        printf("Correctness: PASS\n");
    } else {
        printf("Correctness: FAIL, errors=%d\n", errors);
        free(input);
        free(gx);
        free(gy);
        free(scalar_mag);
        free(rvv_mag);
        return 1;
    }

    // Timing comparison.
    uint64_t scalar_cycles = 0;
    uint64_t rvv_cycles = 0;

    for (int it = 0; it < iterations; ++it) {
        uint64_t t0 = read_cycles();
        gradient_magnitude_l1(gx, gy, scalar_mag, width, height);
        uint64_t t1 = read_cycles();

        gradient_magnitude_l1_rvv(gx, gy, rvv_mag, width, height);
        uint64_t t2 = read_cycles();

        scalar_cycles += (t1 - t0);
        rvv_cycles += (t2 - t1);
    }

    printf("Iterations: %d\n", iterations);
    printf("Average Scalar L1 cycles: %llu\n",
           (unsigned long long)(scalar_cycles / iterations));
    printf("Average RVV L1 cycles: %llu\n",
           (unsigned long long)(rvv_cycles / iterations));

    free(input);
    free(gx);
    free(gy);
    free(scalar_mag);
    free(rvv_mag);

    return 0;
}
