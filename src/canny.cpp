#include "canny.h"
#include <fstream>
#include <iostream>

// دالة تحويل الصورة (أو نسخ البيانات للـ Grayscale)
void grayscale(unsigned char* input, unsigned char* output, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        output[i] = input[i]; // في حالتنا هنا هي عملية نسخ للبكسلات
    }
}

// دالة قراءة الصورة
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

// دالة كتابة الصورة
bool write_raw_image(const char* filename, int width, int height, unsigned char* data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing " << filename << std::endl;
        return false;
    }

    file.write(reinterpret_cast<const char*>(data), width * height);
    
    file.close();
    return true;
}



void write_raw_image(const char* filename, unsigned char* data, int size) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    outfile.write(reinterpret_cast<const char*>(data), size);
    outfile.close();
    std::cout << "Successfully saved: " << filename << std::endl;
}