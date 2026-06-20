#include <iostream>
#include <vector>
#include <chrono> 
#include <cstdint>
#include <iomanip>
#include "canny.h"
#include "gradient.h"
// #include "image_data.h" // 📥 Not needed anymore since we are reading from file!

using namespace std;

int main() 
{
    std::cout << "==================================================\n";
    std::cout << "      Canny Edge Detection: Full Pipeline Profiling \n";
    std::cout << "==================================================\n";

    const int width = 1216;
    const int height = 704;

    // 1. Read the image from file (the function allocates memory for us)
    unsigned char* input_image = nullptr;
    if (!read_raw_image("data/test.raw", width, height, input_image)) {
        std::cerr << "Error: Failed to read data/test.raw. Make sure the file exists!" << std::endl;
        return -1;
    }
    cout << "Image read successfully from data/test.raw!" << std::endl;

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
    
    // بنباصي المصفوفة اللي قرأناها من الملف
    gaussian_blur_2d<uint8_t, int32_t, int16_t>(input_image, blurred_image.data(), width, height);
    
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

    // =================================================================
    // حفظ الصورة النهائية 
    // =================================================================
    // We are saving 'magnitude.data()' which is the final edge image
    write_raw_image("data/output.raw", width, height, magnitude.data());
    cout << "File saved successfully to data/output.raw!" << std::endl;
    
    // Free the dynamically allocated image memory
    delete[] input_image;
    
    return 0;
}
