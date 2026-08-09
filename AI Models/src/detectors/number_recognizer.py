from ultralytics import YOLO
from src.config import NUMBER_CONFIDENCE

class NumberRecognizer:
    def __init__(self, weights_path):
        """Load the jersey number recognition model"""
        print("Loading jersey number recognition model...")
        # Load your custom weights (trained on 0-9)
        self.model = YOLO(weights_path)

    def recognize(self, frame, bbox):
        """
        Receives the frame and player coordinates, crops the image to find the jersey number.
        """
        x1, y1, x2, y2 = bbox
        
        # Crop player image from full frame
        player_crop = frame[y1:y2, x1:x2]
        
        # Check if crop was successful and image is not empty
        if player_crop.size == 0:
            return None
            
        # 1. Pass player crop to number model
        results = self.model(player_crop, conf=NUMBER_CONFIDENCE, verbose=False)
        
        digits = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # أ. جلب إحداثي X للرقم (عشان نعرف مكانه فين على التيشيرت)
                digit_x1 = int(box.xyxy[0][0])
                
                # ب. جلب اسم الكلاس (الرقم نفسه من 0 لـ 9)
                digit_class = int(box.cls[0]) 
                
                # حفظ مكان الرقم وقيمته
                digits.append((digit_x1, str(digit_class)))
                
        # لو ملقاش أي أرقام على التيشيرت في الفريم ده
        if len(digits) == 0:
            # Fallback: لو الباوندري كبير والكاميرا قريبة، جرب تدور في منطقة الشورت
            h_crop = y2 - y1
            if h_crop > 150: # باوندري كبير
                # منطقة الشورت تقريباً من 45% إلى 85% من الطول
                ys1 = int(y1 + 0.45 * h_crop)
                ys2 = int(y1 + 0.85 * h_crop)
                shorts_crop = frame[ys1:ys2, x1:x2]
                
                if shorts_crop.size > 0:
                    shorts_results = self.model(shorts_crop, conf=NUMBER_CONFIDENCE, verbose=False)
                    for result in shorts_results:
                        boxes = result.boxes
                        for box in boxes:
                            digit_x1 = int(box.xyxy[0][0])
                            digit_class = int(box.cls[0]) 
                            digits.append((digit_x1, str(digit_class)))
                            
        # لو ملقاش أي أرقام
        if len(digits) == 0:
            return None
            
        # 2. ترتيب الأرقام من اليسار لليمين بناءً على إحداثي X
        # الخطوة دي مهمة جداً عشان لو الرقم 25، مايتقريش 52
        digits.sort(key=lambda d: d[0])
        
        # 3. دمج الأرقام مع بعض في نص واحد (String)
        # مثلاً: ['1', '0'] هتتحول لـ "10"
        final_number = "".join([d[1] for d in digits])
        
        # تصفية الأرقام غير المنطقية (رقم صفر، أو 3 أرقام فأكثر)
        try:
            val = int(final_number)
            if val == 0 or len(final_number) >= 3:
                return None
        except ValueError:
            return None
            
        return final_number