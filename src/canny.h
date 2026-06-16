#ifndef CANNY_H
#define CANNY_H

#include <cstdint>
#include <cstddef>

// =================================================================
// 1. تعريف الثوابت (يتم تحديد قيمها في ملف canny.cpp)
// =================================================================
extern const int16_t GAUSSIAN_KERNEL_5X5[5][5];
extern const int32_t GAUSSIAN_SUM;

// الثوابت الخاصة بالفلتر المنفصل 1D
extern const int16_t GAUSSIAN_KERNEL_1D[5];
extern const int32_t GAUSSIAN_SUM_1D;

// =================================================================
// 2. تعريف الدوال العادية (ليست Templates - كودها في canny.cpp)
// =================================================================
void grayscale(const unsigned char* input, unsigned char* output, int width, int height);
bool read_raw_image(const char* filename, int width, int height, unsigned char*& data);
bool write_raw_image(const char* filename, int width, int height, const unsigned char* data);

// =================================================================
// 3. دالة التمويه الغاوسي القياسية ثنائية الأبعاد (2D Scalar Blur)
// يجب أن تبقى في الـ Header لأنها تحتوي على جسم الدالة بالكامل
// =================================================================
template <typename PixelT, typename AccumulatorT, typename KernelT>
void gaussian_blur_scalar(const PixelT* input, PixelT* output, size_t width, size_t height) {
    for (size_t r = 0; r < height; ++r) {
        for (size_t c = 0; c < width; ++c) {
            
            // تصحيح: استخدام نوع AccumulatorT المرن بدلاً من int32_t الثابتة
            AccumulatorT accumulator = 0; 

            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    int64_t nr = static_cast<int64_t>(r) + ky;
                    int64_t nc = static_cast<int64_t>(c) + kx;

                    if (nr >= 0 && nr < static_cast<int64_t>(height) &&
                        nc >= 0 && nc < static_cast<int64_t>(width)) {
                        
                        // ضرب القيمة بعد تحويلها لنوع المجمع لضمان عدم حدوث الطفحان (Overflow)
                        accumulator += static_cast<AccumulatorT>(input[nr * width + nc]) * static_cast<AccumulatorT>(GAUSSIAN_KERNEL_5X5[ky + 2][kx + 2]);
                    }
                }
            }

            // تصحيح: التقسيم باستخدام الثابت العام المتوافق مع الـ Template
            AccumulatorT result = accumulator / static_cast<AccumulatorT>(GAUSSIAN_SUM);

            // التأكد من أن القيمة بين 0 و 255 (Clamping)
            if (result > 255) result = 255;
            if (result < 0)   result = 0;

            // تصحيح: تحويل الناتج إلى النوع PixelT المطلوب وليس unsigned char بشكل إجباري
            output[r * width + c] = static_cast<PixelT>(result);
        }
    }
}

// =================================================================
// 4. إعلان دالة التمويه الغاوسي المنفصلة (Separable Scalar Blur)
// هنا نضع الإعلان فقط، والكود الفعلي سيكون في canny.cpp مع الـ Explicit Instantiation
// =================================================================
template <typename PixelT, typename AccumulatorT, typename KernelT>
void gaussian_blur_separable(const PixelT* input, PixelT* output, size_t width, size_t height);

#endif // CANNY_H