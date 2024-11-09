# -*- coding: utf-8 -*-
"""
Load CNN Model for Tomato Leaf Disease Detection, Evaluate, and Predict
"""

# Step 1: Suppress OneDNN Optimizations (Optional)
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Turn off OneDNN to avoid precision warnings

# Step 2: Import Necessary Libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Step 3: Load the Saved Model
try:
    model = load_model(r"C:\Users\Manohar\Desktop\tomatodst\tomato\tomato_leaf_disease_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)  # Stop the script if loading fails



# Step 5: Make a Prediction on a New Image
def predict_image(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Class labels (adjust if needed based on your model)
    class_labels = [
        'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold',
        'Septoria Leaf Spot', 'Spider Mites', 'Target Spot',
        'Yellow Leaf Curl Virus', 'Mosaic Virus', 'Healthy'
    ]

    print(f"Predicted class: {class_labels[predicted_class]}")
    return class_labels[predicted_class]

# Step 6: Test Set Evaluation (Optional)
# If you have a test set prepared, uncomment the following lines:
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_set = test_datagen.flow_from_directory(
#     'val', target_size=(128, 128), batch_size=64, class_mode='categorical'
# )
# evaluate_model(test_set)  # Evaluate model accuracy on the test set

# Step 7: Make Predictions on Test Images
image_path = r"C:\Users\Manohar\Desktop\data tmt\input\20240616_122255.jpg" # Replace with your own image path
predicted_label = predict_image(image_path)

print(f"Final Prediction: {predicted_label}")
