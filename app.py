import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# -------------------- FONCTION POUR CHARGER LES MODÈLES --------------------
@st.cache_resource
def load_model(model_name):
    return tf.keras.models.load_model(model_name)

# -------------------- FONCTION DE PRÉTRAITEMENT --------------------
def preprocess_image(image):
    image = image.convert("L")  # convertir en niveaux de gris
    image = image.resize((128, 128))  # adapter à la taille d'entrée de ton modèle
    image = np.array(image)  # convertir l'image en tableau numpy
    image = image.reshape(1, 128, 128, 1)
    return image

# -------------------- TITRE --------------------
st.title("Classification d'IRM Cérébrales")

# -------------------- SÉLECTION DU MODÈLE --------------------
model_names = {
    "CNN": "cnn_model.h5",
    "InceptionV3": "inception_model.h5",
    "VGG": "vgg_model.h5"
}

model_choice = st.selectbox("Choisissez un modèle :", list(model_names.keys()))
model_path = model_names[model_choice]
model = load_model(model_path)

# -------------------- UPLOAD IMAGE --------------------
uploaded_file = st.file_uploader("Téléversez une image IRM :", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)

    if st.button("Prédire"):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)

        # Affichage des résultats
        class_names = ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]
        predicted_class = class_names[np.argmax(prediction)]
        st.success(f"Classe prédite : {predicted_class}")
