import streamlit as st
import os
from PIL import Image
from cropper_image import YOLOv8Cropper
from data_extractor import OCRExtractor

# Setting environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define the class names that you are interested in
class_names = [
    'Discount_Percentage', 'Due_Date', 'Email_Client', 'Name_Client', 'Products',
    'Remise', 'Subtotal', 'Tax', 'Tax_Percentage', 'Tel_Client', 'billing address',
    'header', 'invoice date', 'invoice number', 'shipping address', 'total'
]

st.title("Invoice Parsing")

# Uploading an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and convert the image to RGB
    image = Image.open(uploaded_file).convert("RGB")

    # Load the model and perform cropping
    model_path = 'invoice_extraction/invoice_model/weights/best.pt'
    cropper = YOLOv8Cropper(model_path=model_path)
    cropped_images = cropper.crop_and_get_images(image)

    # Perform OCR extraction on cropped images
    ocr_extractor = OCRExtractor()
    ocr_results = ocr_extractor.extract_text(cropped_images)

    # Filter OCR results based on class names
    filtered_results = {k: v for k, v in ocr_results.items() if k in class_names}

    # Display extracted data if it exists
    if filtered_results:
        st.write("### Extracted Data:")
        for label, text in filtered_results.items():
            st.write(f"**{label}:**")
            st.write(text)
    else:
        st.write("No relevant data extracted.")

    # Display cropped images with labels
    st.write("### Cropped Images:")
    if cropped_images:
        for i, (cropped_img, label) in enumerate(cropped_images):
            st.image(cropped_img, caption=f"{label} {i}", use_column_width=True)
    else:
        st.write("No objects detected.")
