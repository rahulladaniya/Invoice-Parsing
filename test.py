# Import necessary libraries
import torch
from ultralytics import YOLO
import os

class YOLOv8Evaluator:
    def __init__(self, model_path, data_config_path, device='cuda'):
        """
        Initializes the YOLOv8Evaluator class with model path and configuration.

        Parameters:
        - model_path (str): Path to the trained YOLO model.
        - data_config_path (str): Path to the YAML configuration file.
        - device (str): Device to be used for inference ('cpu' or 'cuda').
        """
        self.model_path = model_path
        self.data_config_path = data_config_path
        self.device = device if torch.cuda.is_available() else 'cpu'  # Automatically switch to 'cpu' if CUDA is unavailable

        # Print device information
        if self.device == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")

        # Check PyTorch CUDA availability
        print("Torch CUDA available:", torch.cuda.is_available())

        # Initialize YOLO model
        self.model = YOLO(self.model_path)

    def run_inference(self, image_path, save_path=None):
        """
        Run inference on an image and optionally save the result.

        Parameters:
        - image_path (str): Path to the image to be processed.
        - save_path (str, optional): Path to save the result image.
        """
        results = self.model(image_path)  # Perform inference
        
        # Iterate over each result
        for result in results:
            result.plot(show=True)  # Display the result
            
            if save_path:
                result.save(save_path)  # Save the result to the specified path

        return results

if __name__ == '__main__':
    # Define paths
    model_path = 'invoice_extraction/invoice_model/weights/best.pt'  # Path to the trained model
    image_path = 'input to test/OIP.jpeg'  # Image to perform inference on
    save_path = 'output/'  # Path to save inference results

    # Initialize evaluator with provided configurations
    evaluator = YOLOv8Evaluator(
        model_path=model_path,
        data_config_path='dataset/data.yaml',  # Path to the data configuration file
        device='cuda'  # Use 'cuda' for GPU, 'cpu' for CPU
    )

    # Run inference
    evaluator.run_inference(image_path, save_path)
