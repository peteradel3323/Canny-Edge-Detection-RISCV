#include "canny.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <riscv_vector.h>

// ========================================================================
// 1. Constants & Macros (Phase 6.2: 2D Convolution with LMUL Sweep)
// ========================================================================

#if GAUSS_LMUL == 4
    typedef vuint8m4_t  vuint8_t;
    typedef vuint16m8_t vuint16_t;
    typedef vuint16m8_t vint32_t;
    typedef vbool2_t    vbool_t;

    #define rvv_vle32 __riscv_vle16_v_u16m8      
    #define rvv_vse32 __riscv_vse16_v_u16m8      
    #define rvv_vadd __riscv_vadd_vv_u16m8
    #define rvv_vmul __riscv_vmul_vx_u16m8
    #define rvv_vdiv __riscv_vdivu_vx_u16m8        // Division for 2D kernel sum
    #define rvv_vmv_v_x __riscv_vmv_v_x_u16m8     // Set vector to scalar (initialization)
    #define rvv_vmax __riscv_vmax_vx_u16m8
    #define rvv_vmin __riscv_vmin_vx_u16m8
    #define rvv_vncvt_16(v, vl) (v)              
    #define rvv_vncvt_8 __riscv_vncvt_x_x_w_u8m4  
    #define rvv_vreinterpret_u8(v) (v)            
    #define rvv_vse8 __riscv_vse8_v_u8m4
    #define VSETVL_E(n) __riscv_vsetvl_e16m8(n)
   
    typedef uint16_t ActualAccumulatorT;

#elif GAUSS_LMUL == 2
    typedef vuint8m2_t  vuint8_t;
    typedef vint16m4_t  vint16_t;
    typedef vint32m8_t  vint32_t;
    typedef vbool4_t    vbool_t;  

    #define rvv_vle32 __riscv_vle32_v_i32m8
    #define rvv_vse32 __riscv_vse32_v_i32m8
    #define rvv_vadd __riscv_vadd_vv_i32m8
    #define rvv_vmul __riscv_vmul_vx_i32m8
    #define rvv_vdiv __riscv_vdiv_vx_i32m8
    #define rvv_vmv_v_x __riscv_vmv_v_x_i32m8
    #define rvv_vmax __riscv_vmax_vx_i32m8
    #define rvv_vmin __riscv_vmin_vx_i32m8
    #define rvv_vncvt_16 __riscv_vncvt_x_x_w_i16m4
    #define rvv_vncvt_8 __riscv_vncvt_x_x_w_i8m2
    #define rvv_vreinterpret_u8(v) __riscv_vreinterpret_v_i8m2_u8m2(v)
    #define rvv_vse8 __riscv_vse8_v_u8m2
    #define VSETVL_E(n) __riscv_vsetvl_e32m8(n)

    typedef int32_t ActualAccumulatorT;

#else // Default GAUSS_LMUL == 1
    typedef vuint8m1_t  vuint8_t;
    typedef vint16m2_t  vint16_t;
    typedef vint32m4_t  vint32_t;
    typedef vbool8_t    vbool_t;  

    #define rvv_vle32 __riscv_vle32_v_i32m4
    #define rvv_vse32 __riscv_vse32_v_i32m4
    #define rvv_vadd __riscv_vadd_vv_i32m4
    #define rvv_vmul __riscv_vmul_vx_i32m4
    #define rvv_vdiv __riscv_vdiv_vx_i32m4
    #define rvv_vmv_v_x __riscv_vmv_v_x_i32m4
    #define rvv_vmax __riscv_vmax_vx_i32m4
    #define rvv_vmin __riscv_vmin_vx_i32m4
    #define rvv_vncvt_16 __riscv_vncvt_x_x_w_i16m2
    #define rvv_vncvt_8 __riscv_vncvt_x_x_w_i8m1
    #define rvv_vreinterpret_u8(v) __riscv_vreinterpret_v_i8m1_u8m1(v)
    #define rvv_vse8 __riscv_vse8_v_u8m1
    #define VSETVL_E(n) __riscv_vsetvl_e32m4(n)

    typedef int32_t ActualAccumulatorT;
#endif

// 2D Matrix Definition
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
// 3. 2D Gaussian Blur RVV Implementation (Phase 6.2 - Vectorized)
// ======================================================================
template <class PixelT, class AccumulatorT, class KernelT>
void gaussian_blur_2d(const PixelT* __restrict input, PixelT* __restrict output, size_t width, size_t height) {
   
    // Calculate padded dimensions to safely apply the 5x5 kernel
    size_t padded_width = width + 4;
    size_t padded_height = height + 4;
   
    // Allocate padded array matching the chosen LMUL Accumulator Type
    ActualAccumulatorT* padded_input = new ActualAccumulatorT[padded_width * padded_height]();

    // Copy original image into the center of the padded array
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            padded_input[(y + 2) * padded_width + (x + 2)] = static_cast<ActualAccumulatorT>(input[y * width + x]);
        }
    }

    // ----------------------------------------------------------------------
    // 2D Convolution Pass -> Fully Vectorized
    // ----------------------------------------------------------------------
    for (size_t y = 2; y < height + 2; ++y) {
        size_t x = 2;
        size_t num_cols = width;
       
        while (num_cols > 0) {
            size_t vl = VSETVL_E(num_cols);
           
            // Initialize the accumulator to zero for the current vector chunk
            vint32_t v_sum = rvv_vmv_v_x(0, vl);
           
            // Loop over the 5x5 Matrix (25 operations)
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    int16_t weight = GAUSSIAN_KERNEL_5X5[ky + 2][kx + 2];
                   
                    // Load adjacent pixels from the padded array
                    vint32_t p = rvv_vle32(reinterpret_cast<const ActualAccumulatorT*>(padded_input + (y + ky) * padded_width + (x + kx)), vl);
                   
                    // Multiply by the weight and accumulate into the sum
                    v_sum = rvv_vadd(v_sum, rvv_vmul(p, weight, vl), vl);
                }
            }

            // Normalize by dividing by the total matrix sum (273)
            vint32_t v_div = rvv_vdiv(v_sum, GAUSSIAN_SUM, vl);

            // Clamp values between 0 and 255 to avoid pixel distortion
            #if GAUSS_LMUL != 4
                vint32_t v_clipped = rvv_vmax(v_div, 0, vl);
                v_clipped = rvv_vmin(v_clipped, 255, vl);
            #else
                vint32_t v_clipped = v_div; // Clamping inherently handled by u16 math
            #endif
           
            // Narrowing the size dynamically back to 8-bit output
            #if GAUSS_LMUL != 4
                vint16_t v16 = rvv_vncvt_16(v_clipped, vl);
                vuint8_t v8 = rvv_vreinterpret_u8(rvv_vncvt_8(v16, vl));
            #else
                vuint8_t v8 = rvv_vncvt_8(v_clipped, vl);
            #endif
           
            // Store the processed pixels back into the original output dimensions
            rvv_vse8(reinterpret_cast<uint8_t*>(output + (y - 2) * width + (x - 2)), v8, vl);
           
            x += vl;
            num_cols -= vl;
        }
    }

    // Free memory to prevent leaks
    delete[] padded_input;
}

// =================================================================
// 4. Explicit Template Instantiation
// =================================================================
template void gaussian_blur_2d<uint8_t, int32_t, int16_t>(
    const uint8_t* __restrict input, uint8_t* __restrict output, size_t width, size_t height
);
template void gaussian_blur_2d<uint8_t, uint16_t, int16_t>(
    const uint8_t* __restrict input, uint8_t* __restrict output, size_t width, size_t height
);