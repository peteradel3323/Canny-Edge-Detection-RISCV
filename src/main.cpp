#include <iostream>
#include "canny.h"

int main() {
    std::cout << "Starting Canny Edge Detection on RISC-V..." << std::endl;
    int width = 64; 
    int height = 64;
    unsigned char* input_image = new unsigned char[width * height];
    unsigned char* output_image = new unsigned char[width * height];

    grayscale(input_image, output_image, width, height);
    
    std::cout << "Grayscale conversion complete!" << std::endl;

    delete[] input_image;
    delete[] output_image;
    return 0;
}