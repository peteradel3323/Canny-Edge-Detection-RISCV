#include "canny.h"
#include <fstream>
#include <iostream>
#include <vector> 

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

const int32_t GAUSSIAN_SUM = 273; 

// الفلتر أحادي الأبعاد (1D) المستخدم في الفلتر المنفصل (Separable)
const int16_t GAUSSIAN_KERNEL_1D[5] = {1, 4, 6, 4, 1};
const int32_t GAUSSIAN_SUM_1D = 16; // مجموع معاملات الفلتر الأحادي = 16

// ==========================================
// 2. تنفيذ دوال معالجة الصور وقراءة/كتابة الملفات
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

// ==========================================
// 3. تنفيذ دالة التمويه الغاوسي المنفصل المحسنة (Separable Blur)
// ==========================================
template <class PixelT, class AccumulatorT, class KernelT>
void gaussian_blur_separable(const PixelT* __restrict input, PixelT* __restrict output, size_t width, size_t height) {
    
    // حساب أبعاد المصفوفات الكبيرة المحشوة بالأصفار (نحتاج +4 بكسل في العرض والارتفاع للفلتر 5x5)
    size_t padded_width = width + 4;
    size_t padded_height = height + 4;
    
    // حجز مصفوفات الحشو وتصفيرها تلقائياً باستخدام () لمنع مشاكل حواف الصورة
    AccumulatorT* padded_input = new AccumulatorT[padded_width * padded_height]();
    AccumulatorT* temp_padded  = new AccumulatorT[padded_width * padded_height]();

    // نسخ الصورة الأصلية إلى منتصف مصفوفة الـ padded_input مع ترك الحواف أصفارًا
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            padded_input[(y + 2) * padded_width + (x + 2)] = static_cast<AccumulatorT>(input[y * width + x]);
        }
    }

    // ----------------------------------------------------------------------
    // التمريرة الأفقية (Horizontal Pass)
    // كود خطي نقي خالي تماماً من لفات الـ k والـ if -> ممتاز للـ Auto-Vectorization
    // ----------------------------------------------------------------------
    for (size_t y = 2; y < height + 2; ++y) {
        for (size_t x = 2; x < width + 2; ++x) {
            AccumulatorT sum = padded_input[y * padded_width + (x - 2)] * static_cast<AccumulatorT>(GAUSSIAN_KERNEL_1D[0])
                             + padded_input[y * padded_width + (x - 1)] * static_cast<AccumulatorT>(GAUSSIAN_KERNEL_1D[1])
                             + padded_input[y * padded_width + (x    )] * static_cast<AccumulatorT>(GAUSSIAN_KERNEL_1D[2])
                             + padded_input[y * padded_width + (x + 1)] * static_cast<AccumulatorT>(GAUSSIAN_KERNEL_1D[3])
                             + padded_input[y * padded_width + (x + 2)] * static_cast<AccumulatorT>(GAUSSIAN_KERNEL_1D[4]);
            
            temp_padded[y * padded_width + x] = sum;
        }
    }

    // ----------------------------------------------------------------------
    // التمريرة الرأسية (Vertical Pass)
    // تم فك اللفة الداخلية وحذف الـ if -> متوازي تماماً للـ Vector Units
    // ----------------------------------------------------------------------
    for (size_t y = 2; y < height + 2; ++y) {
        for (size_t x = 2; x < width + 2; ++x) {
            AccumulatorT sum = temp_padded[(y - 2) * padded_width + x] * static_cast<AccumulatorT>(GAUSSIAN_KERNEL_1D[0])
                             + temp_padded[(y - 1) * padded_width + x] * static_cast<AccumulatorT>(GAUSSIAN_KERNEL_1D[1])
                             + temp_padded[(y    ) * padded_width + x] * static_cast<AccumulatorT>(GAUSSIAN_KERNEL_1D[2])
                             + temp_padded[(y + 1) * padded_width + x] * static_cast<AccumulatorT>(GAUSSIAN_KERNEL_1D[3])
                             + temp_padded[(y + 2) * padded_width + x] * static_cast<AccumulatorT>(GAUSSIAN_KERNEL_1D[4]);
            
            // التطبيع (الحسم الكلي عبر القسمة على المجموع الكلي للتمريرتين 16 * 16 = 256)
            sum /= (GAUSSIAN_SUM_1D * GAUSSIAN_SUM_1D);
            
            // الحصر (Clamping) آمن عتادياً ومسرع بـ Vector Min/Max
            if (sum < 0) sum = 0;
            if (sum > 255) sum = 255;
            
            // إرجاع الخرج إلى مصفوفة الصورة الأصلية ذات الأبعاد الأساسية
            output[(y - 2) * width + (x - 2)] = static_cast<PixelT>(sum);
        }
    }

    // تنظيف الذاكرة المؤقتة لمنع الـ Leak
    delete[] padded_input;
    delete[] temp_padded;
}

// =================================================================
// 4. الاستنساخ الصريح (Explicit Template Instantiation)
// تم إضافة __restrict لتتوافق مع الـ Header تماماً
// =================================================================
template void gaussian_blur_separable<uint8_t, int32_t, int16_t>(
    const uint8_t* __restrict input, uint8_t* __restrict output, size_t width, size_t height
);


