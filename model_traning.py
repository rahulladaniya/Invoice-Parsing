from ultralytics import YOLO
import torch
import os

def main():
    # Define class names and paths
    class_names = [
        'Discount_Percentage', 'Due_Date', 'Email_Client', 'Name_Client', 'Products',
        'Remise', 'Subtotal', 'Tax', 'Tax_Precentage', 'Tel_Client', 'billing address',
        'header', 'invoice date', 'invoice number', 'shipping address', 'total'
    ]
    dataset_path = 'dataset'
    model_path = 'yolov8n.pt'
    config_path = os.path.join(dataset_path, 'yolov8_invoice_config.yaml')

    # Check for GPU availability and set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("Using CPU")

    # Create YAML configuration content
    yaml_content = f"""
path: {dataset_path}
train: train
val: valid
test: test
nc: {len(class_names)}
names: {class_names}
    """

    # Write YAML content to file
    with open(config_path, 'w') as file:
        file.write(yaml_content)

    # Load YOLO model
    model = YOLO(model_path)

    # Train the model
    model.train(
        data=config_path,
        epochs=10,
        batch=4, 
        device='cuda',
        project='invoice_extraction',
        name='invoice_model',
        workers=4,
        augment=True,
        patience=5
    )

    print("Training Completed!")
    
    
if __name__ == "__main__":
    main()