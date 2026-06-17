from PIL import Image
import numpy as np

# اسم الصورة الأصلية الخاصة بك
input_filename = 'data/filter_test.jpeg' # تأكد من وضع صورتك في المجلد بهذا الاسم
output_filename = 'data/filter_test.raw'

# فتح الصورة وتحويلها لـ Grayscale
img = Image.open(input_filename).convert('L') 

# تغيير الأبعاد لتكون مطابقة تماماً (1200x675)
img = img.resize((1216, 704))

# تحويل الصورة إلى مصفوفة (Array) من نوع uint8 (بايت واحد لكل بكسل)
img_data = np.array(img, dtype=np.uint8)

# حفظ البيانات في ملف RAW
img_data.tofile(output_filename)

print(f"تم التحويل بنجاح! حجم الملف الناتج: {img_data.size} بايت")