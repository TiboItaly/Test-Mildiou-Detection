# === Training_Model.py ===
# === Étape 1 : Import des bibliothèques ===
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# === Étape 2 : Définition des chemins ===
train_dir = "images"

# === Étape 3 : Préparation des données ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True,
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False,
)

print("Classes détectées :", train_generator.class_indices)
print("Nombre de classes :", len(train_generator.class_indices))
print("Nombre d'images pour l'entraînement :", train_generator.samples)
print("Nombre d'images pour la validation :", validation_generator.samples)

# === Calcul automatique des class weights ===
# train_generator.classes contient l'étiquette (0/1) pour chaque image d'entraînement
y_train = train_generator.classes
classes = np.unique(y_train)
computed_weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, computed_weights)}
print("Class weights utilisés :", class_weight_dict)

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
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# === Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint("best_modele_detection_feuilles.h5", monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7)
]

# === Étape 5 : Entraînement avec class_weight ===
EPOCHS = 50

history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // validation_generator.batch_size),
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# === Étape 6 : Visualisation ===
plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history.get('accuracy', []), label='Acc (train)')
plt.plot(history.history.get('val_accuracy', []), label='Acc (val)')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history.get('precision', []), label='Precision (train)')
plt.plot(history.history.get('val_precision', []), label='Precision (val)')
plt.plot(history.history.get('recall', []), label='Recall (train)')
plt.plot(history.history.get('val_recall', []), label='Recall (val)')
plt.title('Precision / Recall')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history.get('loss', []), label='Loss (train)')
plt.plot(history.history.get('val_loss', []), label='Loss (val)')
plt.title('Loss')
plt.legend()

plt.show()

# === Étape 7 : Sauvegarde du modèle ===
model.save("modele_detection_feuilles_final.h5")
print("Modèle final sauvegardé ! (Le meilleur modèle est aussi enregistré dans best_modele_detection_feuilles.h5)")