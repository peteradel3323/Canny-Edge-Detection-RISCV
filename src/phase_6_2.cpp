#include "canny.h"
#include <fstream>
#include <iostream>
#include <vector> 

// ==========================================
// 1. تعريف الثوابت والمكروهات (Definitions)
// ==========================================

#include <riscv_vector.h>

// تفعيل التنقل الديناميكي بناءً على فلاج الكومبيلر
#if GAUSS_LMUL == 4
    // مجمع 16-بت بدون إشارة (Unsigned) آمن عتادياً ومسموح به حتى m8
    typedef vuint8m4_t  vuint8_t;
    typedef vuint16m8_t vuint16_t; 
    typedef vuint16m8_t vint32_t; // كاستينج ذكي لعدم تعديل اسم المتغيرات داخل الدالة
    typedef vbool2_t    vbool_t;

    #define rvv_vle32 __riscv_vle16_v_u16m8       
    #define rvv_vse32 __riscv_vse16_v_u16m8       
    #define rvv_vadd __riscv_vadd_vv_u16m8
    #define rvv_vmul __riscv_vmul_vx_u16m8
    #define rvv_vsra __riscv_vsrl_vx_u16m8        // التعديل هنا: استخدام الـ Logical Shift بدلاً من Arithmetic
    #define rvv_vmax __riscv_vmax_vx_u16m8
    #define rvv_vmin __riscv_vmin_vx_u16m8
    #define rvv_vncvt_16(v, vl) (v)               
    #define rvv_vncvt_8 __riscv_vncvt_x_x_w_u8m4  
    #define rvv_vreinterpret_u8(v) (v)            
    #define rvv_vse8 __riscv_vse8_v_u8m4
    
    typedef uint16_t ActualAccumulatorT;
#elif GAUSS_LMUL == 2
    typedef vuint8m2_t  vuint8_t;
    typedef vint16m4_t  vint16_t; 
    typedef vint32m8_t  vint32_t; 
    typedef vbool4_t    vbool_t;  

    #define rvv_vle32 __riscv_vle32_v_i32m8
    #define rvv_vse32 __riscv_vse32_v_i32m8
    #define rvv_vadd __riscv_vadd_vv_i32m8
    #define rvv_vmul __riscv_vmul_vx_i32m8
    #define rvv_vsra __riscv_vsra_vx_i32m8
    #define rvv_vmax __riscv_vmax_vx_i32m8
    #define rvv_vmin __riscv_vmin_vx_i32m8
    #define rvv_vncvt_16 __riscv_vncvt_x_x_w_i16m4
    #define rvv_vncvt_8 __riscv_vncvt_x_x_w_i8m2
    #define rvv_vreinterpret_u8(v) __riscv_vreinterpret_v_i8m2_u8m2(v)
    #define rvv_vse8 __riscv_vse8_v_u8m2

    typedef int32_t ActualAccumulatorT;

#else // الافتراضي GAUSS_LMUL == 1
    typedef vuint8m1_t  vuint8_t;
    typedef vint16m2_t  vint16_t;
    typedef vint32m4_t  vint32_t;
    typedef vbool8_t    vbool_t;  

    #define rvv_vle32 __riscv_vle32_v_i32m4
    #define rvv_vse32 __riscv_vse32_v_i32m4
    #define rvv_vadd __riscv_vadd_vv_i32m4
    #define rvv_vmul __riscv_vmul_vx_i32m4
    #define rvv_vsra __riscv_vsra_vx_i32m4
    #define rvv_vmax __riscv_vmax_vx_i32m4
    #define rvv_vmin __riscv_vmin_vx_i32m4
    #define rvv_vncvt_16 __riscv_vncvt_x_x_w_i16m2
    #define rvv_vncvt_8 __riscv_vncvt_x_x_w_i8m1
    #define rvv_vreinterpret_u8(v) __riscv_vreinterpret_v_i8m1_u8m1(v)
    #define rvv_vse8 __riscv_vse8_v_u8m1

    typedef int32_t ActualAccumulatorT;
#endif

const int16_t GAUSSIAN_KERNEL_5X5[5][5] = {
    {1,  4,  7,  4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1,  4,  7,  4, 1}
};

const int32_t GAUSSIAN_SUM = 273; 

const int16_t GAUSSIAN_KERNEL_1D[5] = {1, 4, 6, 4, 1};
const int32_t GAUSSIAN_SUM_1D = 16; 

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
// 3. تنفيذ دالة التمويه الغاوسي المنفصل المحسنة (Separable Blur) بـ RVV
// ==========================================
template <class PixelT, class AccumulatorT, class KernelT>
void gaussian_blur_separable(const PixelT* __restrict input, PixelT* __restrict output, size_t width, size_t height) {
    
    size_t padded_width = width + 4;
    size_t padded_height = height + 4;
    
    ActualAccumulatorT* padded_input = new ActualAccumulatorT[padded_width * padded_height]();
    ActualAccumulatorT* temp_padded  = new ActualAccumulatorT[padded_width * padded_height]();

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            padded_input[(y + 2) * padded_width + (x + 2)] = static_cast<ActualAccumulatorT>(input[y * width + x]);
        }
    }

    // تجهيز حاسب الـ Vector Length بناءً على الـ LMUL المختار عتادياً
    #if GAUSS_LMUL == 4
        #define VSETVL_E32(n) __riscv_vsetvl_e16m8(n) 
    #elif GAUSS_LMUL == 2
        #define VSETVL_E32(n) __riscv_vsetvl_e32m8(n)
    #else
        #define VSETVL_E32(n) __riscv_vsetvl_e32m4(n)
    #endif

    // ----------------------------------------------------------------------
    // التمريرة الأفقية (Horizontal Pass) -> Vectorized 🚀
    // ----------------------------------------------------------------------
    for (size_t y = 2; y < height + 2; ++y) {
        size_t x = 2;
        size_t num_cols = width;
        
        while (num_cols > 0) {
            size_t vl = VSETVL_E32(num_cols);
            
            // تحميل الـ 5 عناصر الأفقية المتجاورة
            vint32_t l0 = rvv_vle32(reinterpret_cast<const ActualAccumulatorT*>(padded_input + y * padded_width + x - 2), vl);
            vint32_t l1 = rvv_vle32(reinterpret_cast<const ActualAccumulatorT*>(padded_input + y * padded_width + x - 1), vl);
            vint32_t l2 = rvv_vle32(reinterpret_cast<const ActualAccumulatorT*>(padded_input + y * padded_width + x    ), vl);
            vint32_t l3 = rvv_vle32(reinterpret_cast<const ActualAccumulatorT*>(padded_input + y * padded_width + x + 1), vl);
            vint32_t l4 = rvv_vle32(reinterpret_cast<const ActualAccumulatorT*>(padded_input + y * padded_width + x + 2), vl);
            
            // حساب المجموع الموزون طبقاً للـ 1D Kernel: [1, 4, 6, 4, 1]
            vint32_t v_sum = l0; 
            v_sum = rvv_vadd(v_sum, rvv_vmul(l1, 4, vl), vl);
            v_sum = rvv_vadd(v_sum, rvv_vmul(l2, 6, vl), vl);
            v_sum = rvv_vadd(v_sum, rvv_vmul(l3, 4, vl), vl);
            v_sum = rvv_vadd(v_sum, l4, vl); 
            
            // تخزين المجموع المؤقت في temp_padded
            rvv_vse32(reinterpret_cast<ActualAccumulatorT*>(temp_padded + y * padded_width + x), v_sum, vl);
            
            x += vl;
            num_cols -= vl;
        }
    }

    // ----------------------------------------------------------------------
    // التمريرة الرأسية (Vertical Pass) -> Vectorized 🚀
    // ----------------------------------------------------------------------
    for (size_t y = 2; y < height + 2; ++y) {
        size_t x = 2;
        size_t num_cols = width;
        
        while (num_cols > 0) {
            size_t vl = VSETVL_E32(num_cols);
            
            // تحميل الـ 5 عناصر الرأسية المتتابعة متوازياً
            vint32_t v0 = rvv_vle32(reinterpret_cast<const ActualAccumulatorT*>(temp_padded + (y - 2) * padded_width + x), vl);
            vint32_t v1 = rvv_vle32(reinterpret_cast<const ActualAccumulatorT*>(temp_padded + (y - 1) * padded_width + x), vl);
            vint32_t v2 = rvv_vle32(reinterpret_cast<const ActualAccumulatorT*>(temp_padded + y * padded_width + x), vl);
            vint32_t v3 = rvv_vle32(reinterpret_cast<const ActualAccumulatorT*>(temp_padded + (y + 1) * padded_width + x), vl);
            vint32_t v4 = rvv_vle32(reinterpret_cast<const ActualAccumulatorT*>(temp_padded + (y + 2) * padded_width + x), vl);
            
            // حساب المجموع الموزون الرأسي: [1, 4, 6, 4, 1]
            vint32_t v_sum = v0;
            v_sum = rvv_vadd(v_sum, rvv_vmul(v1, 4, vl), vl);
            v_sum = rvv_vadd(v_sum, rvv_vmul(v2, 6, vl), vl);
            v_sum = rvv_vadd(v_sum, rvv_vmul(v3, 4, vl), vl);
            v_sum = rvv_vadd(v_sum, v4, vl);
            
            // القسمة على 256 عن طريق ترحيل الخانات لليمين بمقدار 8 بتات
            vint32_t v_div = rvv_vsra(v_sum, 8, vl);
            
            // حصر الأرقام المفرطة (Clipping) بين 0 و 255 لمنع تشوه البكسلات
            #if GAUSS_LMUL != 4
                vint32_t v_clipped = rvv_vmax(v_div, 0, vl);
                v_clipped = rvv_vmin(v_clipped, 255, vl);
            #else
                vint32_t v_clipped = v_div; // غير مطلوبة في الـ u16 الحسابية لأن الخرج مضمون بين 0 و 255
            #endif
            
            // تقزيم الحجم المتوافق ديناميكياً
            #if GAUSS_LMUL != 4
                vint16_t v16 = rvv_vncvt_16(v_clipped, vl);
                vuint8_t v8 = rvv_vreinterpret_u8(rvv_vncvt_8(v16, vl));
            #else
                vuint8_t v8 = rvv_vncvt_8(v_clipped, vl);
            #endif
            
            // تخزين البكسلات النهائية الصافية في مصفوفة الـ output
            rvv_vse8(reinterpret_cast<uint8_t*>(output + (y - 2) * width + (x - 2)), v8, vl);
            
            x += vl;
            num_cols -= vl;
        }
    }

    #undef VSETVL_E32

    delete[] padded_input;
    delete[] temp_padded;
}

// =================================================================
// 4. الاستنساخ الصريح (Explicit Template Instantiation)
// =================================================================
template void gaussian_blur_separable<uint8_t, int32_t, int16_t>(
    const uint8_t* __restrict input, uint8_t* __restrict output, size_t width, size_t height
);
template void gaussian_blur_separable<uint8_t, uint16_t, int16_t>(
    const uint8_t* __restrict input, uint8_t* __restrict output, size_t width, size_t height
);