import streamlit as st
import os
import gdown
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
file_id = "154M5eBFu3mrwicYwlIkK3V8j7aazLzPT"
path = "resnet_best_model10.keras"

def load_model():
    if not os.path.exists("freshness_best_model.keras"):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)
    return load_model(path)

food_classes = ["apel", "apem", "bakpia", "jeruk", "kue pastel", "lemper", "onde-onde", "putu ayu", "risol", "roti"]

st.title("🍽️ Food Category Detection")

# Upload image
img_file = st.file_uploader("Upload Foto", type=["jpg", "png", "jpeg"])
camera_file = st.camera_input("Atau gunakan Kamera")
model = load_model (path)

if img_file or camera_file:
    img = Image.open(img_file or camera_file).convert("RGB")
    st.image(img, caption="Gambar Input", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224,224))
    x = np.expand_dims(np.array(img_resized)/255.0, axis=0)

    # Predict
    food_pred = np.argmax(model.predict(x), axis=1)[0]

    st.subheader("🔎 Hasil Prediksi")
    st.write("🍔 Food Category:", food_classes[food_pred])
