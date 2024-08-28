import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class YOLOv8Predictor:
    def __init__(self, model_path, device='cuda'):
        """
        Initializes the YOLOv8Predictor class with model path and device.

        Parameters:
        - model_path (str): Path to the trained YOLO model.
        - device (str): Device to be used for inference ('cpu' or 'cuda').
        """
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else 'cpu'  # Automatically switch to 'cpu' if CUDA is unavailable
        self.model = YOLO(self.model_path)  # Load the YOLO model

    def predict(self, image_path):
        """
        Performs inference on the given image and returns annotated image.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - annotated_image (PIL Image): Image with bounding boxes drawn.
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)

        # Perform inference
        results = self.model(image_array)
        # Results might be a list of results, so handle accordingly
        if isinstance(results, list) and len(results) > 0:
            results = results[0]

        # Extract bounding boxes and labels from results
        annotated_img = results.plot()  # Use plot method to draw annotations

        # Convert annotated image to PIL Image
        annotated_image = Image.fromarray(annotated_img)

        return annotated_image

    def display_image(self, image):
        """
        Displays the image using matplotlib.

        Parameters:
        - image (PIL Image): Image to be displayed.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def save_image(self, image, output_path):
        """
        Saves the image to the specified path.

        Parameters:
        - image (PIL Image): Image to be saved.
        - output_path (str): Path to save the image.
        """
        image.save(output_path)

if __name__ == '__main__':
    # Path to the YOLOv8 model
    model_path = 'invoice_extraction/invoice_model/weights/best.pt'  # Update this path to your trained model
    image_path = 'input to test/Total-Due-Amount.jpg'  # Path to the new image for prediction
    output_path = 'output'  # Path to save the annotated image

    # Initialize the predictor
    predictor = YOLOv8Predictor(model_path=model_path)

    # Perform prediction
    annotated_image = predictor.predict(image_path)

    # Display and save the annotated image
    predictor.display_image(annotated_image)
    predictor.save_image(annotated_image, output_path)

    print(f"Annotated image saved to {output_path}")
