import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import cv2
from joblib import load
from skimage.filters import gabor
import mahotas as mt

# Load models and scalers
@st.cache_resource
def load_cnn_model():
    cnn_model_path = r"C:\Users\Manohar\Desktop\Tomato disease app\tomato_leaf_disease_model.h5"
    return load_model(cnn_model_path)

@st.cache_resource
def load_svm_model_and_scaler():
    svm_model_path = r"C:\Users\Manohar\Desktop\Tomato disease app\svm_model_best.pkl"
    scaler_path = r"C:\Users\Manohar\Desktop\Tomato disease app\scaler.pkl"
    return load(svm_model_path), load(scaler_path)

# Define class labels for CNN model
cnn_class_labels = {
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

# Define class labels for SVM model
svm_disease_mapping = {
    0: "Tomato___healthy",
    1: "Tomato___Bacterial_spot",
    2: "Tomato___Early_blight",
    3: "Tomato___Late_blight",
    4: "Tomato___Leaf_Mold",
    5: "Tomato___Septoria_leaf_spot",
    6: "Tomato___Spider_mites_Two-spotted_spider_mite",
    7: "Tomato___Target_Spot",
    8: "Tomato___Tomato_mosaic_virus",
    9: "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
}

# Preprocess the image for SVM model
def preprocess_image_for_svm(image):
    """
    Extract 23 features from the image for SVM model prediction.
    """
    radius = 3
    no_points = 8 * radius

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaborFilt_real, gaborFilt_imag = gabor(gray, frequency=0.6)
    gaborFilt = (gaborFilt_real**2 + gaborFilt_imag**2) // 2
    gabor_hist, _ = np.histogram(gaborFilt, bins=8)
    gabor_hist = np.array(gabor_hist, dtype=float)
    gabor_prob = np.divide(gabor_hist, np.sum(gabor_hist))
    gabor_energy = np.sum(gabor_prob**2)
    gabor_entropy = -np.sum(gabor_prob * np.log2(gabor_prob + 1e-10))

    blur = cv2.GaussianBlur(gray, (25, 25), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")
    cnt = contours[0]

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    circularity = (perimeter**2) / area if area > 0 else 0
    equi_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0

    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)
    red_var = np.std(red_channel)
    green_var = np.std(green_channel)
    blue_var = np.std(blue_channel)

    textures = mt.features.haralick(gray)
    ht_mean = textures.mean(axis=0)

    return np.array([
        gabor_energy, gabor_entropy, w, h, area, rect_area, perimeter,
        aspect_ratio, extent, solidity, hull_area, circularity,
        equi_diameter, red_mean, green_mean, blue_mean,
        red_var, green_var, blue_var, ht_mean[1], ht_mean[2], ht_mean[4], ht_mean[8]
    ])

# Streamlit app
def main():
    st.title("Tomato Leaf Disease Detection by Jithina,  Manohar,  Yusra  ")
    st.write("Choose a model to detect diseases in tomato leaves.")

    # Model selection
    model_choice = st.radio("Select Model", ["CNN Model", "SVM Model"])

    # Upload image
    uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        image = np.array(Image.open(uploaded_file))

        if model_choice == "CNN Model":
            cnn_model = load_cnn_model()
            target_height, target_width, _ = cnn_model.input_shape[1:4]
            image_resized = Image.fromarray(image).resize((target_width, target_height))
            image_array = img_to_array(image_resized) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            predictions = cnn_model.predict(image_array)
            predicted_class_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_index]
            predicted_class_name = cnn_class_labels.get(predicted_class_index, "Unknown Class")

            st.success(f"Prediction: {predicted_class_name}")
            st.info(f"Confidence: {confidence:.2f}")
            st.bar_chart(predictions[0])

        elif model_choice == "SVM Model":
            svm_model, scaler = load_svm_model_and_scaler()
            try:
                feature_vector = preprocess_image_for_svm(image)
                scaled_features = scaler.transform([feature_vector])
                prediction = svm_model.predict(scaled_features)[0]
                decision_scores = svm_model.decision_function(scaled_features)
                confidence = np.max(decision_scores) / np.sum(np.abs(decision_scores)) * 100

                disease = svm_disease_mapping.get(prediction, "Unknown Disease")
                st.success(f"Prediction: {disease}")
                st.info(f"Confidence Level: {confidence:.2f}%")
            except Exception as e:
                st.error(f"Error processing the image: {e}")

if __name__ == "__main__":
    main()
