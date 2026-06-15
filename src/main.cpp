#include <iostream>
#include "canny.h"

using namespace std;

int main() {
    cout << "Starting Canny Edge Detection on RISC-V..." << std::endl;
    
    int width = 1200; 
    int height = 675;
    
    // حجز مساحة في الذاكرة
    unsigned char* input_image = new unsigned char[width * height];
    unsigned char* output_image = new unsigned char[width * height];

    // 1. قراءة الملف أولاً (خطوة ضرورية جداً!)
    read_raw_image("data/test.raw", width, height , input_image);
    cout << "Image read successfully!" << std::endl;

    // 2. التحويل (المعالجة)
    grayscale(input_image, output_image, width, height);
    cout << "Grayscale conversion complete!" << std::endl;

    // 3. الكتابة (استخدام output_image اللي عليه النتيجة)
    write_raw_image("data/output.raw", output_image, width * height);
    cout << "File saved successfully!" << std::endl;


    if (!read_raw_image("data/test.raw", width, height, input_image)) 
    {
        std::cerr << "image can not be rode" << std::endl;
        return -1; // الخروج من البرنامج في حالة الخطأ
    }

    // تنظيف الذاكرة
    delete[] input_image;
    delete[] output_image;
    
    return 0;
}