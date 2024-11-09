import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the model
@st.cache_resource
def load_disease_model():
    model_path = r"C:\Users\Manohar\Desktop\tomatodst\tomato\tomato_leaf_disease_model.h5"  # Update with your model's path
    model = load_model(model_path)
    return model

# Define class labels
class_labels = {
    0: 'Bacterial Spot',
    1: 'Early Blight',
    2: 'Late Blight',
    3: 'Leaf Mold',
    4: 'Septoria Leaf Spot',
    5: 'Spider Mites',
    6: 'Target Spot',
    7: 'Yellow Leaf Curl Virus',
    8: 'Mosaic Virus',
    9: 'Healthy'
}

# Main Streamlit app
def main():
    st.title("Tomato Leaf Disease Detection")
    st.write("Upload a tomato leaf image to predict the disease.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Load and preprocess the image
        model = load_disease_model()
        expected_input_shape = model.input_shape

        # Preprocess the uploaded image
        image = Image.open(uploaded_file)
        if len(expected_input_shape) == 4:  # Model expects (batch_size, height, width, channels)
            target_height, target_width, target_channels = expected_input_shape[1:4]
            image = image.resize((target_width, target_height))
            image_array = img_to_array(image) / 255.0  # Normalize
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        else:
            st.error("Unsupported model input shape. Please check the model.")
            return

        # Make predictions
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]
        predicted_class_name = class_labels.get(predicted_class_index, "Unknown Class")

        # Display prediction results
        st.success(f"Predicted Class: {predicted_class_name}")
        st.info(f"Confidence: {confidence:.2f}")

        # Visualize prediction confidence
        st.bar_chart(predictions[0])

if __name__ == "__main__":
    main()
