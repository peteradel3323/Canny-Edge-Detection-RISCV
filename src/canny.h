#ifndef CANNY_H
#define CANNY_H

void grayscale(unsigned char* input, unsigned char* output, int width, int height);

// دالة قراءة الصور (Raw Grayscale)
bool read_raw_image(const char* filename, int width, int height, unsigned char*& data);

// دالة كتابة الصور
bool write_raw_image(const char* filename, int width, int height, unsigned char* data);


void write_raw_image(const char* filename, unsigned char* data, int size);

#endif