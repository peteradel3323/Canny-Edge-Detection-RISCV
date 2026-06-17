#include <iostream>
#include <vector>
#include <chrono> // مكتبة مهمة جداً لقياس الوقت
#include <cstdint>
#include "canny.h"

int main() 
{
    std::cout << "--- Canny Edge Detection Testbench ---" << std::endl;

    // أبعاد الصورة الاختبارية (يمكنك تغييرها لاحقاً لتطابق صورتك الحقيقية)
    const int width = 1216;
    const int height = 704;
    const char* input_filename = "data/filter_test.raw"; // ضع صورتك بهذا الاسم بجانب ملف التشغيل

    unsigned char* input_image = nullptr;

    // 1. محاولة قراءة صورة من ملف
    if (read_raw_image(input_filename, width, height, input_image)) {
        std::cout << "Successfully loaded image: " << input_filename << std::endl;
    } else {
        // إذا لم يجد الصورة، سنصنع صورة اختبارية (مربع أبيض في وسط خلفية سوداء)
        std::cout << "Warning: Could not load " << input_filename << ". Generating synthetic test image..." << std::endl;
        input_image = new unsigned char[width * height];
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // رسم مربع أبيض في المنتصف
                if (x > 200 && x < 312 && y > 200 && y < 312) {
                    input_image[y * width + x] = 255; 
                } else {
                    input_image[y * width + x] = 0;
                }
            }
        }
    }

    // تجهيز مصفوفات لتخزين الصور الناتجة
    std::vector<uint8_t> output_2d(width * height, 0);
    std::vector<uint8_t> output_separable(width * height, 0);

    // =================================================================
    // 2. اختبار الفلتر العادي (2D Gaussian Blur) وقياس وقته
    // =================================================================
    std::cout << "\nRunning 2D Gaussian Blur..." << std::endl;
    auto start_2d = std::chrono::high_resolution_clock::now();
    
    // استدعاء الدالة مع تحديد أنواع القوالب (Templates)
    gaussian_blur_scalar<uint8_t, int32_t, int16_t>(input_image, output_2d.data(), width, height);
    
    auto end_2d = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_2d = end_2d - start_2d;
    std::cout << "-> 2D Blur execution time: " << duration_2d.count() << " ms" << std::endl;

    // =================================================================
    // 3. اختبار الفلتر المنفصل (Separable Gaussian Blur) وقياس وقته
    // =================================================================
    std::cout << "\nRunning Separable Gaussian Blur..." << std::endl;
    auto start_sep = std::chrono::high_resolution_clock::now();
    
    gaussian_blur_separable<uint8_t, int32_t, int16_t>(input_image, output_separable.data(), width, height);
    
    auto end_sep = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_sep = end_sep - start_sep;
    std::cout << "-> Separable Blur execution time: " << duration_sep.count() << " ms" << std::endl;

    // =================================================================
    // 4. حفظ النتائج في ملفات للمعاينة
    // =================================================================
    write_raw_image("data/output_2d_blur.raw", width, height, output_2d.data());
    write_raw_image("data/output_separable_blur.raw", width, height, output_separable.data());

    std::cout << "\nImages saved successfully. Check 'output_2d_blur.raw' and 'output_separable_blur.raw'." << std::endl;

    // تنظيف الذاكرة
    delete[] input_image;

    return 0;
}