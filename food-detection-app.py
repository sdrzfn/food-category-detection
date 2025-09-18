import streamlit as st
import os
import gdown
import numpy as np
import tensorflow as tf
from PIL import Image

# --- Config ---
FILE_ID = "154M5eBFu3mrwicYwlIkK3V8j7aazLzPT"
MODEL_PATH = "resnet_best_model.tflite"
IMG_SIZE = (224, 224)

food_classes = ["apel", "apem", "bakpia", "jeruk", "kue pastel", 
                "lemper", "onde-onde", "putu ayu", "risol", "roti"]

# --- Load TFLite Model ---
@st.cache_resource
def load_tflite_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Preprocessing ---
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

# --- Streamlit UI ---
st.title("🍽️ Food Category Detection")

img_file = st.file_uploader("Upload Foto", type=["jpg", "png", "jpeg"])
camera_file = st.camera_input("Atau gunakan Kamera")

if img_file or camera_file:
    img = Image.open(img_file or camera_file).convert("RGB")
    st.image(img, caption="Gambar Input", use_column_width=True)

    # Preprocess
    x = preprocess_image(img)

    # Predict
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    food_pred = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][food_pred]

    st.subheader("🔎 Hasil Prediksi")
    st.write("🍔 Food Category:", food_classes[food_pred])
    st.write(f"📊 Confidence: {confidence*100:.2f}%")
