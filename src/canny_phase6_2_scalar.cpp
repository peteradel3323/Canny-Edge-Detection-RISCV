#include "canny.h"
#include <fstream>
#include <iostream>
#include <vector> 

// ========================================================================
// 1. Constants & Type Definitions (Phase 6.2: 2D Scalar Convolution)
// ========================================================================

// استخدام النوع العادي Accumulator لأننا قمنا بإلغاء ماكروز المتجهات
typedef int32_t ActualAccumulatorT;

// 2D Matrix (Matching your provided definitions)
const int16_t GAUSSIAN_KERNEL_5X5[5][5] = {
    {1,  4,  7,  4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1,  4,  7,  4, 1}
};

const int32_t GAUSSIAN_SUM = 273; 

// ==========================================
// 2. Image Processing & File I/O Functions
// ==========================================

void grayscale(const unsigned char* input, unsigned char* output, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        output[i] = input[i]; 
    }
}

bool read_raw_image(const char* filename, int width, int height, unsigned char*& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    data = new unsigned char[width * height];
    file.read(reinterpret_cast<char*>(data), width * height);
    
    file.close();
    return true;
}

bool write_raw_image(const char* filename, int width, int height, const unsigned char* data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing " << filename << std::endl;
        return false;
    }

    file.write(reinterpret_cast<const char*>(data), width * height);
    file.close();
    return true;
}

// ======================================================================
// 3. 2D Gaussian Blur SCALAR Implementation (Phase 6.2 - No Vectors)
// ======================================================================
template <class PixelT, class AccumulatorT, class KernelT>
void gaussian_blur_separable(const PixelT* __restrict input, PixelT* __restrict output, size_t width, size_t height) {
    
    // Calculate padded dimensions to safely apply the 5x5 kernel
    size_t padded_width = width + 4;
    size_t padded_height = height + 4;
    
    // Allocate padded array
    ActualAccumulatorT* padded_input = new ActualAccumulatorT[padded_width * padded_height]();

    // Copy original image into the center of the padded array
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            padded_input[(y + 2) * padded_width + (x + 2)] = static_cast<ActualAccumulatorT>(input[y * width + x]);
        }
    }

    // ----------------------------------------------------------------------
    // 2D Convolution Pass -> Pure Scalar (Pixel by Pixel)
    // ----------------------------------------------------------------------
    for (size_t y = 2; y < height + 2; ++y) {
        for (size_t x = 2; x < width + 2; ++x) {
            
            ActualAccumulatorT sum = 0;
            
            // Loop over the 5x5 Matrix (25 scalar operations per pixel)
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    int16_t weight = GAUSSIAN_KERNEL_5X5[ky + 2][kx + 2];
                    
                    // Load adjacent pixel and multiply by weight
                    ActualAccumulatorT pixel = padded_input[(y + ky) * padded_width + (x + kx)];
                    sum += pixel * weight;
                }
            }
            
            // Normalize by dividing by the total matrix sum (273)
            ActualAccumulatorT div = sum / GAUSSIAN_SUM;
            
            // Clamp values between 0 and 255 to avoid pixel distortion
            if (div < 0)   div = 0;
            if (div > 255) div = 255;
            
            // Store the processed pixel back into the original output dimensions
            output[(y - 2) * width + (x - 2)] = static_cast<PixelT>(div);
        }
    }

    // Free memory to prevent leaks
    delete[] padded_input;
}

// =================================================================
// 4. Explicit Template Instantiation
// =================================================================
template void gaussian_blur_separable<uint8_t, int32_t, int16_t>(
    const uint8_t* __restrict input, uint8_t* __restrict output, size_t width, size_t height
);
template void gaussian_blur_separable<uint8_t, uint16_t, int16_t>(
    const uint8_t* __restrict input, uint8_t* __restrict output, size_t width, size_t height
);