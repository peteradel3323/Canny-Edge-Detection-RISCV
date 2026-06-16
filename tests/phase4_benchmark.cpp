#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#include "gradient.h"

static inline uint64_t read_cycles() {
    uint64_t cycles;
    asm volatile ("rdcycle %0" : "=r"(cycles));
    return cycles;
}

static void generate_test_image(uint8_t* img, int width, int height) {
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
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
    const int width = 1200;
    const int height = 675;
    const int n = width * height;
    const int iterations = 100;

    uint8_t* input = (uint8_t*)malloc(n);
    int16_t* gx = (int16_t*)malloc(n * sizeof(int16_t));
    int16_t* gy = (int16_t*)malloc(n * sizeof(int16_t));
    uint8_t* mag_l1 = (uint8_t*)malloc(n);
    uint8_t* mag_l2 = (uint8_t*)malloc(n);
    uint8_t* dir = (uint8_t*)malloc(n);

    if (!input || !gx || !gy || !mag_l1 || !mag_l2 || !dir) {
        printf("Allocation failed\n");
        return 1;
    }

    generate_test_image(input, width, height);

    compute_sobel(input, gx, gy, width, height);
    gradient_magnitude_l1(gx, gy, mag_l1, width, height);
    gradient_magnitude_l2(gx, gy, mag_l2, width, height);
    gradient_direction(gx, gy, dir, width, height);

    uint64_t sobel_cycles = 0;
    uint64_t mag_l1_cycles = 0;
    uint64_t mag_l2_cycles = 0;
    uint64_t direction_cycles = 0;

    uint64_t total_start = read_cycles();

    for (int i = 0; i < iterations; ++i) {
        uint64_t t0 = read_cycles();
        compute_sobel(input, gx, gy, width, height);
        uint64_t t1 = read_cycles();

        gradient_magnitude_l1(gx, gy, mag_l1, width, height);
        uint64_t t2 = read_cycles();

        gradient_magnitude_l2(gx, gy, mag_l2, width, height);
        uint64_t t3 = read_cycles();

        gradient_direction(gx, gy, dir, width, height);
        uint64_t t4 = read_cycles();

        sobel_cycles += (t1 - t0);
        mag_l1_cycles += (t2 - t1);
        mag_l2_cycles += (t3 - t2);
        direction_cycles += (t4 - t3);
    }

    uint64_t total_end = read_cycles();

    uint64_t checksum =
        checksum_u8(mag_l1, n) +
        checksum_u8(mag_l2, n) +
        checksum_u8(dir, n);

    printf("Image size: %d x %d\n", width, height);
    printf("Iterations: %d\n", iterations);
    printf("Average Sobel cycles: %llu\n", (unsigned long long)(sobel_cycles / iterations));
    printf("Average Magnitude L1 cycles: %llu\n", (unsigned long long)(mag_l1_cycles / iterations));
    printf("Average Magnitude L2 cycles: %llu\n", (unsigned long long)(mag_l2_cycles / iterations));
    printf("Average Direction cycles: %llu\n", (unsigned long long)(direction_cycles / iterations));
    printf("Average Total cycles: %llu\n", (unsigned long long)((total_end - total_start) / iterations));
    printf("Checksum: %llu\n", (unsigned long long)checksum);

    free(input);
    free(gx);
    free(gy);
    free(mag_l1);
    free(mag_l2);
    free(dir);

    return 0;
}
