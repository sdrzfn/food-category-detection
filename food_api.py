from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load trained food category model
model = tf.keras.models.load_model("resnet_best_model10.tflite")

# Define class labels sesuai urutan training
food_labels = [
    "Apel", "Apem", "Bakpia", "Jeruk", "Kue Pastel", 
    "Lemper", "Onde-Onde", "Putu Ayu", "Risol", "Roti"
]

@app.get("/")
async def root():
    return {"message": "Food Category Detection API is running!"}

@app.post("/category")
async def predict(file: UploadFile = File(...)):
    # Baca file gambar
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))

    # Preprocess ke format model
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # [1,224,224,3]

    # Prediksi
    prediction = model.predict(img_array)[0]  # [n_classes]
    max_idx = np.argmax(prediction)
    confidence = float(prediction[max_idx])

    return {
        "label": food_labels[max_idx],
        "confidence": confidence,
        "all_predictions": {food_labels[i]: float(prediction[i]) for i in range(len(food_labels))}
    }
