from PIL import Image, ImageFilter
import numpy as np
import torch
from ultralytics import YOLO
import os
import shutil

class YOLOv8Cropper:
    output_dir = 'cropped_images'
    
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.model_path)
        
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        # print(f"Cleared output directory: {self.output_dir}")

    # def enhance_image(self, image):
    #     enhanced_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    #     return enhanced_image

    def crop_and_get_images(self, image):
        # image = self.enhance_image(image)
        
        image_array = np.array(image)
        results = self.model(image_array)

        cropped_images = []
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            label = results[0].names[int(box.cls[0])]
            cropped_img = image.crop((x1, y1, x2, y2))
            cropped_images.append((cropped_img, label))
            cropped_img_path = os.path.join(self.output_dir, f"{label}_{i}.png")
            cropped_img.save(cropped_img_path)
        return cropped_images
