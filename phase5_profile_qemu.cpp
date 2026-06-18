#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>

#include "gradient.h"
#include "embedded_image.h"

static constexpr int NUM_ITERATIONS = 100;
using Clock = std::chrono::high_resolution_clock;

static double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) / 2.0 : v[n/2];
}

// Self-contained 5x5 Gaussian blur (sigma~1.0, sum=273), zero-padding boundary
static const int16_t GAUSSIAN_5X5[5][5] = {
    {1, 4, 7, 4, 1},
    {4,16,26,16, 4},
    {7,26,41,26, 7},
    {4,16,26,16, 4},
    {1, 4, 7, 4, 1}
};
static const int32_t GAUSSIAN_SUM = 273;

static void gaussian_blur_5x5(const uint8_t* input, uint8_t* output, int width, int height) {
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int32_t acc = 0;
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    int nr = r + ky;
                    int nc = c + kx;
                    uint8_t pixel = 0;
                    if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
                        pixel = input[nr * width + nc];
                    }
                    acc += static_cast<int32_t>(pixel) * GAUSSIAN_5X5[ky + 2][kx + 2];
                }
            }
            int32_t result = acc / GAUSSIAN_SUM;
            if (result > 255) result = 255;
            if (result < 0)   result = 0;
            output[r * width + c] = static_cast<uint8_t>(result);
        }
    }
}

static double time_gaussian_stage(const uint8_t* input, uint8_t* output, int w, int h) {
    std::vector<double> samples;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto t0 = Clock::now();
        gaussian_blur_5x5(input, output, w, h);
        auto t1 = Clock::now();
        samples.push_back(ms_between(t0, t1));
    }
    return median(samples);
}

static double time_sobel_stage(const uint8_t* blurred, int16_t* Gx, int16_t* Gy, int w, int h) {
    std::vector<double> samples;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto t0 = Clock::now();
        compute_sobel(blurred, Gx, Gy, w, h);
        auto t1 = Clock::now();
        samples.push_back(ms_between(t0, t1));
    }
    return median(samples);
}

static double time_magnitude_stage(const int16_t* Gx, const int16_t* Gy, uint8_t* mag, int w, int h) {
    std::vector<double> samples;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto t0 = Clock::now();
        gradient_magnitude_l1(Gx, Gy, mag, w, h);
        auto t1 = Clock::now();
        samples.push_back(ms_between(t0, t1));
    }
    return median(samples);
}

static double time_magnitude_l2_stage(const int16_t* Gx, const int16_t* Gy, uint8_t* mag, int w, int h) {
    std::vector<double> samples;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto t0 = Clock::now();
        gradient_magnitude_l2(Gx, Gy, mag, w, h);
        auto t1 = Clock::now();
        samples.push_back(ms_between(t0, t1));
    }
    return median(samples);
}

static double time_direction_stage(const int16_t* Gx, const int16_t* Gy, uint8_t* dir, int w, int h) {
    std::vector<double> samples;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto t0 = Clock::now();
        gradient_direction(Gx, Gy, dir, w, h);
        auto t1 = Clock::now();
        samples.push_back(ms_between(t0, t1));
    }
    return median(samples);
}

int main(int argc, char** argv) {
    int W = (argc >= 2) ? atoi(argv[1]) : 548;
    int H = (argc >= 3) ? atoi(argv[2]) : 342;
    int n = W * H;

    printf("=== Phase 5: Pipeline Profiling (QEMU, embedded image) ===\n");
    printf("Image: %dx%d, %d iterations per stage\n\n", W, H, NUM_ITERATIONS);

    if (EMBEDDED_IMAGE_SIZE != n) {
        printf("ERROR: embedded image size %d does not match %dx%d=%d\n", EMBEDDED_IMAGE_SIZE, W, H, n);
        return 1;
    }

    uint8_t* input = (uint8_t*)aligned_alloc(64, n);
    memcpy(input, EMBEDDED_IMAGE, n);

    uint8_t* blurred  = (uint8_t*)aligned_alloc(64, n);
    int16_t* Gx       = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
    int16_t* Gy       = (int16_t*)aligned_alloc(64, n * sizeof(int16_t));
    uint8_t* mag_l1   = (uint8_t*)aligned_alloc(64, n);
    uint8_t* mag_l2   = (uint8_t*)aligned_alloc(64, n);
    uint8_t* dir      = (uint8_t*)aligned_alloc(64, n);

    gaussian_blur_5x5(input, blurred, W, H);
    compute_sobel(blurred, Gx, Gy, W, H);

    printf("Timing Gaussian blur...\n");
    double t_gaussian = time_gaussian_stage(input, blurred, W, H);

    printf("Timing Sobel gradient (Gx, Gy)...\n");
    double t_sobel = time_sobel_stage(blurred, Gx, Gy, W, H);

    printf("Timing Magnitude (L1)...\n");
    double t_magnitude_l1 = time_magnitude_stage(Gx, Gy, mag_l1, W, H);

    printf("Timing Magnitude (L2)...\n");
    double t_magnitude_l2 = time_magnitude_l2_stage(Gx, Gy, mag_l2, W, H);

    printf("Timing Direction...\n");
    double t_direction = time_direction_stage(Gx, Gy, dir, W, H);

    double total = t_gaussian + t_sobel + t_magnitude_l1 + t_direction;

    printf("\n=== Per-Stage Timing (median of %d runs) ===\n", NUM_ITERATIONS);
    printf("Gaussian      %.4f ms  %.1f%%\n", t_gaussian, 100.0*t_gaussian/total);
    printf("Sobel Gx/Gy   %.4f ms  %.1f%%\n", t_sobel, 100.0*t_sobel/total);
    printf("Magnitude L1  %.4f ms  %.1f%%\n", t_magnitude_l1, 100.0*t_magnitude_l1/total);
    printf("Magnitude L2  %.4f ms  (extra)\n", t_magnitude_l2);
    printf("Direction     %.4f ms  %.1f%%\n", t_direction, 100.0*t_direction/total);
    printf("TOTAL         %.4f ms  100.0%%\n", total);

    printf("\nDone.\n");

    free(input); free(blurred); free(Gx); free(Gy);
    free(mag_l1); free(mag_l2); free(dir);
    return 0;
}
