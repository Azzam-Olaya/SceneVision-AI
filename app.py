from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


st.set_page_config(page_title="SceneVision AI", page_icon="🌍")

PROJECT_DIR = Path(__file__).parent
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMAGE_SIZE = (150, 150)
SELECTED_MODEL = PROJECT_DIR / "models" / "cnn_scenevision_selected.keras"
BASE_MODEL = PROJECT_DIR / "models" / "cnn_scenevision.keras"
MODEL_PATH = SELECTED_MODEL if SELECTED_MODEL.exists() else BASE_MODEL
METRICS_PATH = PROJECT_DIR / "results" / "selected_model_test_metrics.csv"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(image):
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    probabilities = load_model().predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
    predicted_class = CLASSES[int(np.argmax(probabilities))]
    return predicted_class, probabilities


st.title("🌍 SceneVision AI")
st.write("Importez une image pour reconnaître le type de scène.")
st.caption("Classes : buildings · forest · glacier · mountain · sea · street")

if not MODEL_PATH.exists():
    st.error("Le modèle est introuvable dans le dossier models.")
    st.stop()

if METRICS_PATH.exists():
    metrics = pd.read_csv(METRICS_PATH, index_col=0)
    test_accuracy = float(metrics.loc["accuracy", "precision"])
    st.metric("Accuracy sur le jeu de test", f"{test_accuracy:.1%}")

uploaded_image = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Image importée", width="stretch")

    predicted_class, probabilities = predict_image(image)
    confidence = float(np.max(probabilities))
    st.success(f"Classe prédite : **{predicted_class}**")
    st.write(f"Score associé : **{confidence:.1%}**")

    scores = pd.DataFrame({"Score": probabilities}, index=CLASSES)
    st.subheader("Scores des six classes")
    st.bar_chart(scores)
    st.caption("Le score indique la préférence du modèle ; il ne garantit pas que la prédiction est correcte.")
