import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess the image for the model.
    """
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

def classify_image(model_path, image_path):
    """
    Classify the image using the given model.
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Preprocess the image
    img_array = load_and_preprocess_image(image_path)

    # Make predictions
    predictions = model.predict(img_array)
    confidence = predictions[0][0]  # Assuming binary classification with one output neuron

    # Determine the class
    if confidence > 0.5:
        predicted_class = "Forest Fire"
        confidence = confidence * 100  # Convert to percentage
    else:
        predicted_class = "No Forest Fire"
        confidence = (1 - confidence) * 100  # Convert to percentage

    # Print the results
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    # Example usage
    model_path = "Forest_fire_detection.keras"  # Path to the model
    image_path = input("Enter the path to the image: ")  # Get image path from user
    classify_image(model_path, image_path)