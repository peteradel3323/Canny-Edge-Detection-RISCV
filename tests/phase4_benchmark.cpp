#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>

#include "gradient.h"

// -----------------------------------------------------------------------------
// Phase 4 benchmark timer
//
// Uses C++ std::chrono::steady_clock to measure wall-clock time in milliseconds.
// This requires compiling Phase 4 with riscv64-linux-gnu-g++,
// not riscv64-unknown-elf-g++.
// -----------------------------------------------------------------------------
static inline double now_ms()
{
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();

    return std::chrono::duration<double, std::milli>(duration).count();
}

// -----------------------------------------------------------------------------
// Generate deterministic grayscale test image.
// Left half is darker and right half is brighter.
// This creates a strong vertical edge for Sobel detection.
// -----------------------------------------------------------------------------
void generate_test_image(uint8_t* img, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            if (c < width / 2) {
                img[r * width + c] =
                    static_cast<uint8_t>((r + c) % 128);
            } else {
                img[r * width + c] =
                    static_cast<uint8_t>(128 + ((r + c) % 128));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Checksum used to verify that all optimization levels produce the same output.
// -----------------------------------------------------------------------------
uint64_t checksum_u8(const uint8_t* data, int n)
{
    uint64_t s = 0;

    for (int i = 0; i < n; ++i) {
        s += data[i];
    }

    return s;
}

int main()
{
    const int width = 1200;
    const int height = 675;
    const int n = width * height;
    const int iterations = 100;

    uint8_t* input = new uint8_t[n];

    int16_t* gx = new int16_t[n];
    int16_t* gy = new int16_t[n];

    uint8_t* mag_l1 = new uint8_t[n];
    uint8_t* mag_l2 = new uint8_t[n];
    uint8_t* dir = new uint8_t[n];

    generate_test_image(input, width, height);

    // Warm-up run before timing.
    compute_sobel(input, gx, gy, width, height);
    gradient_magnitude_l1(gx, gy, mag_l1, width, height);
    gradient_magnitude_l2(gx, gy, mag_l2, width, height);
    gradient_direction(gx, gy, dir, width, height);

    double sobel_total_ms = 0.0;
    double mag_l1_total_ms = 0.0;
    double mag_l2_total_ms = 0.0;
    double dir_total_ms = 0.0;

    for (int it = 0; it < iterations; ++it) {
        double t0 = now_ms();

        compute_sobel(input, gx, gy, width, height);

        double t1 = now_ms();

        gradient_magnitude_l1(gx, gy, mag_l1, width, height);

        double t2 = now_ms();

        gradient_magnitude_l2(gx, gy, mag_l2, width, height);

        double t3 = now_ms();

        gradient_direction(gx, gy, dir, width, height);

        double t4 = now_ms();

        sobel_total_ms += (t1 - t0);
        mag_l1_total_ms += (t2 - t1);
        mag_l2_total_ms += (t3 - t2);
        dir_total_ms += (t4 - t3);
    }

    const double avg_sobel_ms = sobel_total_ms / iterations;
    const double avg_mag_l1_ms = mag_l1_total_ms / iterations;
    const double avg_mag_l2_ms = mag_l2_total_ms / iterations;
    const double avg_dir_ms = dir_total_ms / iterations;

    const double avg_total_ms =
        avg_sobel_ms +
        avg_mag_l1_ms +
        avg_mag_l2_ms +
        avg_dir_ms;

    const uint64_t checksum =
        checksum_u8(mag_l1, n) +
        checksum_u8(mag_l2, n) +
        checksum_u8(dir, n);

    std::printf("Image size: %d x %d\n", width, height);
    std::printf("Iterations: %d\n", iterations);

    std::printf("Average Sobel ms: %.6f\n", avg_sobel_ms);
    std::printf("Average Magnitude L1 ms: %.6f\n", avg_mag_l1_ms);
    std::printf("Average Magnitude L2 ms: %.6f\n", avg_mag_l2_ms);
    std::printf("Average Direction ms: %.6f\n", avg_dir_ms);
    std::printf("Average Total ms: %.6f\n", avg_total_ms);

    std::printf("Checksum: %llu\n",
                static_cast<unsigned long long>(checksum));

    delete[] input;
    delete[] gx;
    delete[] gy;
    delete[] mag_l1;
    delete[] mag_l2;
    delete[] dir;

    return 0;
}
