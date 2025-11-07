# Training_Model.py

# === Étape 1 : Import des bibliothèques ===
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# === Étape 2 : Définition des chemins ===
train_dir = "images"

# === Étape 3 : Préparation des données ===
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalisation des pixels, Redimensionner les images à 128x128 pixels
    validation_split=0.2,  # 20% des données pour la validation
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

#Redimensionner les images à 128x128 pixels.
#Normaliser les valeurs des pixels (entre 0 et 1).
train_generator = datagen.flow_from_directory(
    train_dir,              # Chemin vers le dossier principal
    target_size=(128, 128), # Redimensionne les images à 128x128 pixels
    batch_size=32,          # Nombre d'images par lot
    class_mode='binary',    # Classification binaire (saines/contaminées)
    subset='training',      # Ensemble d'entraînement
).repeat()

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation',  # Ensemble de validation
)

# Affiche les classes détectées
print("Classes détectées :", train_generator.class_indices)
print("Nombre de classes :", len(train_generator.class_indices))

print("Nombre d'images par classe :", train_generator.samples)


# === Étape 4 : Construction du modèle ===
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),                   # Pour éviter le surapprentissage
    Dense(1, activation='sigmoid')  # Sortie binaire (0 = saine, 1 = contaminée)
])

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# === Étape 5 : Entraînement ===
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,      # Nombre d'époques (vous pouvez augmenter si nécessaire)
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
)

# === Étape 6 : Visualisation ===
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Précision (Entraînement)')
plt.plot(history.history['val_accuracy'], label='Précision (Validation)')
plt.title('Précision')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perte (Entraînement)')
plt.plot(history.history['val_loss'], label='Perte (Validation)')
plt.title('Perte')
plt.legend()
plt.show()

# === Étape 7 : Sauvegarde du modèle ===
model.save("modele_detection_feuilles.h5")
print("Modèle sauvegardé !")



