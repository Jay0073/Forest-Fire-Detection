import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Function to load and preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    # Convert image to RGB if it has an alpha channel
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize image to target size
    image = image.resize(target_size)
    # Convert image to array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('Forest_fire_detection.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Sidebar with model information
st.sidebar.title("Model Information")
st.sidebar.markdown("""
### Forest Fire Detection Model
- **Model Name**: Forest_fire_detection.keras
- **Type**: Convolutional Neural Network (CNN)
- **Input**: Images resized to 224x224 pixels, RGB format
- **Output**: Binary classification (Forest Fire or No Forest Fire)
- **Training Data**: Trained on a dataset of forest images with and without fire (specific dataset details not provided)
- **Purpose**: Detects the presence of fire in forest images for early warning systems
- **Usage**: Upload a JPEG or PNG image to classify it. Ensure the image is clear and relevant to forest environments.
""")

# Main app
st.title("Forest Fire Detection App")
st.write("Upload an image to classify whether it depicts a forest fire or not.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = Image.open(uploaded_file)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Resize image for display
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        # Preprocess the image
        processed_image = preprocess_image(image)

        # Load the model
        model = load_model()
        
        if model is not None:
            # Make prediction
            try:
                prediction = model.predict(processed_image)
                confidence = prediction[0][0]  # Assuming sigmoid output for binary classification
                
                # Determine class and confidence
                if confidence > 0.5:
                    class_label = "No Forest Fire"
                    confidence_percent = confidence * 100
                else:
                    class_label = "Forest Fire"
                    confidence_percent = (1 - confidence) * 100

                # Display results
                st.subheader("Classification Result")
                st.write(f"**Predicted Class**: {class_label}")
                st.write(f"**Confidence**: {confidence_percent:.2f}%")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload an image to proceed.")