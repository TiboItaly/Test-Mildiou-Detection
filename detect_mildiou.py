import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

print("=== Début du script ===")  # ← 1. Premier print

def detect_mildiou(image_path):
    print(f"Chargement de l'image : {image_path}")  # ← 2. Debug du chemin
    # Charge l'image
    img = cv2.imread(image_path)
    if img is None: # ← Vérifie si l'image est chargée
        print("Erreur : Image non trouvée")
        return
    
    print(f"Image chargée (taille : {img.shape})")  # ← 3. Confirmation

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convertit en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ajoute un flou gaussien avant le seuil
    gray = cv2.GaussianBlur(gray, (5,5),0)
    
    # applique un seuil pour détecter les taches sombres
    # _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # utiliser la méthode d’Otsu pour un seuil automatique :
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Trouve les contours des taches
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrage des petits contours
    min_area = 100
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # Dessine les contours sur l'image originale
    result = img_rgb.copy()
    cv2.drawContours(result, filtered_contours, -1, (0, 255, 0), 2)

    print("Traitement terminé, affichage...")  # ← 4. Avant l'affichage


    # Affiche les résultats
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image originale")
    plt.imshow(img_rgb)
    plt.subplot(1, 2, 2)
    plt.title("Détection de mildiou (taches)")
    plt.imshow(result)
    plt.show()
    print("Fenêtre Matplotlib ouverte. Appuie sur Entrée pour fermer.")
    input()  # Attend une confirmation

if __name__ == "__main__":
    # Remplace par le chemin de ton image (ex. : "data/leaf.jpg")
    detect_mildiou("leaf.jpg")
    print("=== Fin du script ===")  # ← 5. Dernier print
