#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono> // Required for steady_clock
#include "canny.h"

int main() {
    const int width = 1216;
    const int height = 704;
    const int iterations = 100; // Running 100 iterations for stable averages
    
    
    uint8_t* input = (uint8_t*)aligned_alloc(64, width * height);
    uint8_t* output = (uint8_t*)aligned_alloc(64, width * height);

    // Initialize input data with dummy non-zero values so the checksum isn't just 0
    for (int i = 0; i < width * height; ++i) {
        input[i] = (uint8_t)(i % 256);
    }

    // Warm-up run
    gaussian_blur_separable<uint8_t, int32_t, int16_t>(input, output, width, height);

    // Start wall-clock timer
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i) {
        gaussian_blur_separable<uint8_t, int32_t, int16_t>(input, output, width, height);
    }

    // Stop wall-clock timer
    auto end = std::chrono::steady_clock::now();

    // Compute checksum over the output array to guarantee the compiler won't skip the loops
    uint32_t checksum = 0;
    for (int i = 0; i < width * height; ++i) {
        checksum += output[i];
    }

    // Calculate duration in milliseconds
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    printf("Average time per Gaussian Blur iteration: %.3f ms\n", duration_ms.count() / iterations);
    printf("Output Checksum: %u\n", checksum);

    free(input);
    free(output);
    return 0;
}