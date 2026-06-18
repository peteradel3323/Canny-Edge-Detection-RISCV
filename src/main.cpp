#include <iostream>
#include <vector>
#include <chrono> 
#include <cstdint>
#include <iomanip>
#include "canny.h"
#include "gradient.h"
#include "image_data.h" // 📥 إدراج مصفوفة الصورة المدمجة هنا فوراً

int main() 
{
    std::cout << "==================================================\n";
    std::cout << "      Canny Edge Detection: Full Pipeline Profiling \n";
    std::cout << "==================================================\n";

    const int width = 1216;
    const int height = 704;

    std::cout << "Successfully loaded embedded image from memory (Flash/ROData).\n";

    // حجز مصفوفات الـ Pipeline
    std::vector<uint8_t> blurred_image(width * height, 0);
    std::vector<int16_t> grad_x(width * height, 0);
    std::vector<int16_t> grad_y(width * height, 0);
    std::vector<uint8_t> magnitude(width * height, 0);
    std::vector<uint8_t> direction(width * height, 0);

    // =================================================================
    // STAGE 1: Gaussian Blur (المرحلة المحسنة بتاعتنا)
    // =================================================================
    auto start_gauss = std::chrono::high_resolution_clock::now();
    
    // بنباصي المصفوفة المدمجة مباشرة input_image_embedded
    gaussian_blur_separable<uint8_t, int32_t, int16_t>(input_image_embedded, blurred_image.data(), width, height);
    
    auto end_gauss = std::chrono::high_resolution_clock::now();
    double t_gauss = std::chrono::duration<double, std::milli>(end_gauss - start_gauss).count();

    // =================================================================
    // STAGE 2: Sobel Filter (كود أصحابك)
    // =================================================================
    auto start_sobel = std::chrono::high_resolution_clock::now();
    
    compute_sobel(blurred_image.data(), grad_x.data(), grad_y.data(), width, height);
    
    auto end_sobel = std::chrono::high_resolution_clock::now();
    double t_sobel = std::chrono::duration<double, std::milli>(end_sobel - start_sobel).count();

    // =================================================================
    // STAGE 3: Magnitude Calculation (كود أصحابك - L1)
    // =================================================================
    auto start_mag = std::chrono::high_resolution_clock::now();
    
    gradient_magnitude_l1(grad_x.data(), grad_y.data(), magnitude.data(), width, height);
    
    auto end_mag = std::chrono::high_resolution_clock::now();
    double t_mag = std::chrono::duration<double, std::milli>(end_mag - start_mag).count();

    // =================================================================
    // STAGE 4: Direction/Orientation (كود أصحابك)
    // =================================================================
    auto start_dir = std::chrono::high_resolution_clock::now();
    
    gradient_direction(grad_x.data(), grad_y.data(), direction.data(), width, height);
    
    auto end_dir = std::chrono::high_resolution_clock::now();
    double t_dir = std::chrono::duration<double, std::milli>(end_dir - start_dir).count();

    // =================================================================
    // طباعة التقرير الشامل لـ Phase 5
    // =================================================================
    double t_total = t_gauss + t_sobel + t_mag + t_dir;
    
    std::cout << "\n>>> PHASE 5: FULL PIPELINE PROFILING BREAKDOWN <<<\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "--------------------------------------------------\n";
    std::cout << "1. Gaussian Blur Time : " << t_gauss << " ms (" << (t_gauss/t_total)*100.0 << "%)\n";
    std::cout << "2. Sobel Filter Time  : " << t_sobel << " ms (" << (t_sobel/t_total)*100.0 << "%)\n";
    std::cout << "3. Magnitude Time     : " << t_mag   << " ms (" << (t_mag/t_total)*100.0   << "%)\n";
    std::cout << "4. Direction Time     : " << t_dir   << " ms (" << (t_dir/t_total)*100.0   << "%)\n";
    std::cout << "--------------------------------------------------\n";
    std::cout << "Total Pipeline Time   : " << t_total << " ms\n";
    std::cout << "==================================================\n";

    // ملحوظة: دوال الـ كتابة (write_raw_image) قد لا تعمل بسبب قيود الـ Bare-metal toolchain 
    // لكن الـ Profiling والـ Execution شغالين وهيطبعوا الأرقام 100% وهو المطلوب للتسليم.
    
    return 0;
}