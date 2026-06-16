#include "canny.h"
#include <fstream>
#include <iostream>
#include <vector> // ضروري من أجل std::vector في الفلتر المنفصل

// ==========================================
// 1. تعريف الثوابت (Definitions)
// ==========================================

// الفلتر ثنائي الأبعاد (2D)
const int16_t GAUSSIAN_KERNEL_5X5[5][5] = {
    {1,  4,  7,  4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1,  4,  7,  4, 1}
};

// تم التصحيح: مجموع الفلتر السابق هو 273 وليس 16!
const int32_t GAUSSIAN_SUM = 273; 

// الفلتر أحادي الأبعاد (1D) المستخدم في الفلتر المنفصل (Separable)
const int16_t GAUSSIAN_KERNEL_1D[5] = {1, 4, 6, 4, 1};
const int32_t GAUSSIAN_SUM_1D = 256; // 16 * 16

// ==========================================
// 2. تنفيذ دوال معالجة الصور وقراءة/كتابة الملفات
// ==========================================

// تم إضافة const للمدخلات ليتطابق مع الـ Header
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

// تم إضافة const للـ data ليتطابق مع الـ Header
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

// نسخة أخرى من دالة الكتابة (Overload)
void write_raw_image(const char* filename, const unsigned char* data, int size) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<const char*>(data), size);
    outfile.close();
    std::cout << "Successfully saved: " << filename << std::endl;
}

// ==========================================
// 3. تنفيذ دالة التمويه الغاوسي المنفصل (Separable Blur)
// ==========================================
template <class PixelT, class AccumulatorT, class KernelT>
void gaussian_blur_separable(const PixelT* input, PixelT* output, size_t width, size_t height) {
    
    // مصفوفة وسيطة لتخزين ناتج التمريرة الأفقية
    std::vector<AccumulatorT> temp(width * height, 0);

    // ------------------------------------------
    // التمريرة الأفقية (Horizontal Pass)
    // ------------------------------------------
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            AccumulatorT sum = 0;
            for (int k = -2; k <= 2; ++k) {
                int nx = static_cast<int>(x) + k;
                
                // حشوة الأصفار (Zero-padding)
                if (nx >= 0 && nx < static_cast<int>(width)) {
                    sum += static_cast<AccumulatorT>(input[y * width + nx]) * GAUSSIAN_KERNEL_1D[k + 2];
                }
            }
            temp[y * width + x] = sum;
        }
    }

    // ------------------------------------------
    // التمريرة الرأسية (Vertical Pass)
    // ------------------------------------------
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            AccumulatorT sum = 0;
            for (int k = -2; k <= 2; ++k) {
                int ny = static_cast<int>(y) + k;
                
                // حشوة الأصفار
                if (ny >= 0 && ny < static_cast<int>(height)) {
                    sum += temp[ny * width + x] * GAUSSIAN_KERNEL_1D[k + 2];
                }
            }
            
            // التطبيع (Normalization)
            sum /= GAUSSIAN_SUM_1D;
            
            // الحصر (Clamping) لتجنب الطفحان
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            
            output[y * width + x] = static_cast<PixelT>(sum);
        }
    }
}

// =================================================================
// 4. الاستنساخ الصريح (Explicit Template Instantiation)
// هذا السطر يمنع خطأ "Undefined Reference" عند الربط (Linking)
// =================================================================
template void gaussian_blur_separable<uint8_t, int32_t, int16_t>(
    const uint8_t* input, uint8_t* output, size_t width, size_t height
);