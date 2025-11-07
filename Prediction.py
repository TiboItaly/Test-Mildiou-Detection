# prediction.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# === Test sur une nouvelle image ===
model = load_model("modele_detection_feuilles.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Faire la prédiction
    prediction = model.predict(img_array)
    
    # Afficher l'image
    plt.imshow(img)
    plt.axis('off')  # Masquer les axes

    # Ajouter le texte de prédiction sur l'image
    if prediction[0] < 0.5:
        print(f"Feuille saine (confiance : {1 - prediction[0][0]:.2f})")
    else:
        print(f"Feuille contaminée (confiance : {prediction[0][0]:.2f})")

    # Afficher l'image avec la prédiction
    plt.show()

# Remplacez par le chemin de votre nouvelle image
predict_image("feuille saine 2.jpg")
