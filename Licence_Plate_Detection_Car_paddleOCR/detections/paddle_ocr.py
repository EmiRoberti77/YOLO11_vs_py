from paddleocr import PaddleOCR
import cv2

# Initialize the OCR model with angle classification ON
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Set everything at init

# Use the new method without cls=
results = ocr.predict('label.jpeg')  # Just pass image path here

# Parse results
for line in results[0]:
    box = line[0]
    text, score = line[1]
    print(f"Text: {text}, Confidence: {score:.2f}, Box: {box}")

