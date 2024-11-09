from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load and preprocess the image
image = load_img('test_image.jpg', target_size=(128, 128))  # Replace with your image path
image = img_to_array(image) / 255.0  # Normalize the image
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Make a prediction
prediction = model.predict(image)
predicted_class = np.argmax(prediction)

# Map the prediction to the class name
class_labels = [
    'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 
    'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 
    'Yellow Leaf Curl Virus', 'Mosaic Virus', 'Healthy'
]

print(f"Predicted class: {class_labels[predicted_class]}")
