import streamlit as st
import cv2
import numpy as np
from joblib import load
from skimage.filters import gabor
import mahotas as mt
from skimage.feature import local_binary_pattern
import pandas as pd

# Load trained SVM model and scaler
model = load(r"C:\Users\Manohar\Desktop\Tomato leaf disease using SVM\svm_model_best.pkl")  # Update with your model's path
scaler = load(r"C:\Users\Manohar\Desktop\Tomato leaf disease using SVM\scaler.pkl")  # Update with your scaler's path



# Folder-to-prediction mapping
disease_mapping = {
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

# Function to preprocess the uploaded image
def preprocess_image(image):
    """
    Preprocess the uploaded image to extract exactly 23 features for the SVM model.
    """
    radius = 3
    no_points = 8 * radius

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gabor filter
    gaborFilt_real, gaborFilt_imag = gabor(gray, frequency=0.6)
    gaborFilt = (gaborFilt_real**2 + gaborFilt_imag**2) // 2
    gabor_hist, _ = np.histogram(gaborFilt, bins=8)
    gabor_hist = np.array(gabor_hist, dtype=float)
    gabor_prob = np.divide(gabor_hist, np.sum(gabor_hist))
    gabor_energy = np.sum(gabor_prob**2)
    gabor_entropy = -np.sum(gabor_prob * np.log2(gabor_prob + 1e-10))  # Avoid log(0)

    # Smoothing
    blur = cv2.GaussianBlur(gray, (25, 25), 0)

    # Thresholding
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")
    cnt = contours[0]

    # Shape-based features
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # Circularity and rectangularity
    if area > 0:
        circularity = (perimeter**2) / area
    else:
        circularity = 0

    # Equivalent diameter
    equi_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0

    # Color features
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)
    red_var = np.std(red_channel)
    green_var = np.std(green_channel)
    blue_var = np.std(blue_channel)

    # Texture features
    textures = mt.features.haralick(gray)
    ht_mean = textures.mean(axis=0)

    # Combine features into a single vector
    feature_vector = [
        gabor_energy, gabor_entropy, w, h, area, rect_area, perimeter,
        aspect_ratio, extent, solidity, hull_area, circularity,
        equi_diameter, red_mean, green_mean, blue_mean,
        red_var, green_var, blue_var, ht_mean[1], ht_mean[2], ht_mean[4], ht_mean[8]
    ]

    return np.array(feature_vector)

# Streamlit UI
st.title("Tomato Leaf Disease Detection by Jithina,  Manohar,  Yusra")

# Upload image
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Uploaded Image")

    try:
        # Preprocess the image
        feature_vector = preprocess_image(image)
        st.write("Extracted Features (Raw):", feature_vector)

        # Scale the features
        scaled_features = scaler.transform([feature_vector])
        st.write("Scaled Features:", scaled_features)

        # Predict using the model
        prediction = model.predict(scaled_features)[0]
        decision_scores = model.decision_function(scaled_features)
        confidence = np.max(decision_scores) / np.sum(np.abs(decision_scores)) * 100

        # Map the prediction to disease name
        disease = disease_mapping.get(prediction, "Unknown Disease")

        # Display prediction and confidence
        st.success(f"Prediction: {disease}")
        st.info(f"Confidence Level: {confidence:.2f}%")

    except Exception as e:
        st.error(f"Error processing the image: {e}")

    # Debugging outputs for analysis
    debug_data = pd.DataFrame([feature_vector], columns=[
        "gabor_energy", "gabor_entropy", "width", "height", "area", "rect_area", "perimeter",
        "aspect_ratio", "extent", "solidity", "hull_area", "circularity",
        "equi_diameter", "red_mean", "green_mean", "blue_mean",
        "red_var", "green_var", "blue_var", "contrast", "correlation", "inverse_diff_moments", "entropy"
    ])
    st.write("Feature DataFrame for Debugging:")
    st.dataframe(debug_data)
