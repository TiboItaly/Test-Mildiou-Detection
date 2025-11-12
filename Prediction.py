# prediction.py
"""
Prediction helper that loads the trained model, preprocesses an image (local path or URL),
and prints the predicted probability with an optional debug mode that shows internal tensors.

Usage:
  python Prediction.py --image <path_or_url> [--debug]

This script mirrors the training preprocessing (resize to 128x128 and rescale /255).
"""

import argparse
import io
import urllib.request
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model


# --- Charger le modèle entraîné ---
MODEL_PATH = "modele_detection_feuilles_final.h5"
model = load_model(MODEL_PATH)


def load_image(img_path, target_size=(128, 128)):
    """Charge une image depuis un chemin local ou une URL et la redimensionne en RGB."""
    if isinstance(img_path, str) and img_path.startswith("http"):
        with urllib.request.urlopen(img_path, timeout=10) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert('RGB')
    else:
        img = Image.open(img_path).convert('RGB')

    img = img.resize(target_size)
    return img


def preprocess_image(pil_img):
    """Convertit une PIL.Image en batch numpy float32 normalisé [0,1]."""
    arr = np.array(pil_img, dtype=np.float32)
    # shape (H, W, 3)
    # Normalisation identique à ImageDataGenerator(rescale=1./255)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


def segment_leaf_rgb(pil_img, debug=False):
    """Tentative simple de segmentation sur la composante verte (RGB).
    Retourne une PIL.Image recadrée sur la bbox de la plus grande zone verte.
    Si aucun masque significatif n'est trouvé, renvoie l'image d'origine.
    Cette méthode évite d'ajouter OpenCV et marche raisonnablement pour feuilles bien visibles.
    """
    arr = np.array(pil_img.convert('RGB'), dtype=np.float32) / 255.0
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Simple critère: le vert doit être supérieur aux autres canaux par un facteur
    mask = (g > r * 1.05) & (g > b * 1.05) & (g > 0.15)

    if debug:
        print("segment mask sum:", int(mask.sum()), "pixels")

    if mask.sum() < 50:
        # trop petit -> on considère que la segmentation a échoué
        if debug:
            print("Segmentation non significative, retour de l'image originale")
        return pil_img

    # trouver bbox
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # ajouter un petit padding
    h, w = arr.shape[:2]
    pad_y = int(0.05 * (y1 - y0 + 1))
    pad_x = int(0.05 * (x1 - x0 + 1))
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)

    cropped = pil_img.crop((x0, y0, x1 + 1, y1 + 1))
    return cropped


def apply_tta_and_predict(pil_img, do_tta=False, debug=False):
    """Applique une petite TTA (original + hflip) et renvoie la moyenne des probabilités.
    Retourne (prob_mean, probs_list)
    """
    imgs = [pil_img]
    if do_tta:
        imgs.append(pil_img.transpose(Image.FLIP_LEFT_RIGHT))

    probs = []
    for im in imgs:
        inp = preprocess_image(im.resize((128, 128)))
        p = model.predict(inp)
        try:
            pr = float(np.ravel(p)[0])
        except Exception:
            pr = float(p[0][0])
        probs.append(pr)
        if debug:
            print("TTA: prob for augment:", pr)

    mean_prob = float(np.mean(probs))
    return mean_prob, probs


def make_gradcam_heatmap(model, pil_img, layer_name=None, debug=False):
    """Compute Grad-CAM heatmap for a single PIL image.
    Returns a heatmap array resized to the PIL image size (values 0..1).
    If layer_name is None, finds the last conv layer automatically.
    """
    # prepare input (model expects 128x128)
    img_resized = pil_img.resize((128, 128))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = tf.expand_dims(img_array, axis=0)

    # find target conv layer name if not provided
    target_layer_name = None
    if layer_name:
        # try to resolve the provided name in the main model first
        try:
            _ = model.get_layer(layer_name)
            target_layer_name = layer_name
        except Exception:
            # search nested models for the layer name
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model):
                    try:
                        _ = layer.get_layer(layer_name)
                        target_layer_name = layer_name
                        break
                    except Exception:
                        continue
    else:
        # scan layers in reverse to find a conv-like layer
        for layer in reversed(model.layers):
            try:
                if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                    # ensure this layer is actually part of the model graph
                    try:
                        _ = model.get_layer(layer.name)
                        # check output shape
                        if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                            target_layer_name = layer.name
                            break
                    except Exception:
                        continue
            except Exception:
                continue

    if target_layer_name is None:
        raise ValueError("No convolutional layer found for Grad-CAM")

    if debug:
        print("Using Grad-CAM layer:", target_layer_name)

    # Try a safe manual forward pass (works for Sequential-like models) to obtain
    # the conv layer activations and the final prediction in the same graph.
    try:
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            x = img_tensor
            conv_outputs = None
            for layer in model.layers:
                x = layer(x)
                if layer.name == target_layer_name:
                    conv_outputs = x
            predictions = x
            # For binary sigmoid, take the scalar output
            loss = predictions[:, 0]
    except Exception:
        # fallback to building a sub-model (older approach)
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(target_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, 0]

    # gradients of the target class w.r.t conv outputs
    grads = tape.gradient(loss, conv_outputs)
    # compute guided weights
    # remove batch dim
    conv_outputs = conv_outputs[0]
    grads = grads[0]

    # global average pooling on the gradients
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # weighted sum of feature maps
    cam = tf.zeros(conv_outputs.shape[:2], dtype=tf.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = tf.nn.relu(cam)
    cam = cam.numpy()

    # normalize
    if cam.max() != 0:
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

    # resize heatmap to original PIL image size
    heatmap = tf.image.resize(cam[..., np.newaxis], (pil_img.size[1], pil_img.size[0]))
    heatmap = tf.squeeze(heatmap).numpy()
    return heatmap


def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.4, cmap='jet'):
    """Overlay a heatmap (2D array 0..1) on a PIL image and return a PIL image."""
    # Use pyplot.get_cmap to avoid Matplotlib deprecation warnings
    heatmap_uint8 = np.uint8(255 * heatmap)
    colormap = plt.get_cmap(cmap)
    colored = colormap(heatmap_uint8 / 255.0)[:, :, :3]
    colored_img = np.uint8(255 * colored)

    img_arr = np.array(pil_img.convert('RGB'), dtype=np.uint8)
    overlay = np.uint8(img_arr * (1 - alpha) + colored_img * alpha)
    return Image.fromarray(overlay)


def predict_image(img_path, debug=False, do_segment=False, do_tta=False, save_path=None, threshold=0.5):
    img = load_image(img_path)

    # Optionnel: segmentation simple pour recadrer sur la feuille
    processed_for_display = img
    if do_segment:
        seg = segment_leaf_rgb(img, debug=debug)
        # garder l'image segmentée pour affichage et prédiction
        img = seg
        processed_for_display = seg

    # Pré-diagnostic des entrées (après recadrage mais avant resize)
    inp_preview = preprocess_image(img.resize((128, 128)))
    if debug:
        print("--- DEBUG: input tensor ---")
        print("input shape:", inp_preview.shape)
        print("dtype:", inp_preview.dtype)
        print("min, max, mean:", float(inp_preview.min()), float(inp_preview.max()), float(inp_preview.mean()))

    # Faire la prédiction (avec TTA si demandé)
    if do_tta:
        prob, probs_list = apply_tta_and_predict(img, do_tta=True, debug=debug)
        if debug:
            print("TTA probs:", probs_list)
    else:
        # prédiction simple
        inp = preprocess_image(img.resize((128, 128)))
        prediction = model.predict(inp)
        try:
            prob = float(np.ravel(prediction)[0])
        except Exception:
            prob = float(prediction[0][0])
        if debug:
            print("model output (raw):", prediction)
            print(f"probability: {prob:.4f}")

    # Interprétation (seuil configurable)
    if prob < threshold:
        label = "Feuille saine"
        confidence = 1.0 - prob
    else:
        label = "Feuille contaminée"
        confidence = prob

    print(f"{label} (confiance : {confidence:.2f}, prob: {prob:.4f})")

    # Afficher l'image
    # Grad-CAM option: compute and overlay heatmap
    if hasattr(predict_image, '_gradcam_request') and predict_image._gradcam_request:
        try:
            heatmap = make_gradcam_heatmap(model, img, layer_name=predict_image._gradcam_layer, debug=debug)
            overlay = overlay_heatmap_on_image(img, heatmap)
            if save_path:
                overlay.save(save_path)
                if debug:
                    print("Saved gradcam overlay to:", save_path)
            else:
                plt.imshow(overlay)
                plt.title(f"{label} ({confidence:.2f}) - GradCAM")
                plt.axis('off')
                plt.show()
        except Exception as e:
            print("Grad-CAM failed:", e)
    else:
        plt.imshow(img)
        plt.title(f"{label} ({confidence:.2f})")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            if debug:
                print("Saved output image to:", save_path)
        else:
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, default="_rvi321_prod_viti_fiche_arinarnoa_niv1_web.jpg",
                        help='Chemin local ou URL de l\'image à prédire')
    parser.add_argument('--debug', action='store_true', help='Afficher les informations de debug')
    parser.add_argument('--segment', action='store_true', help='Appliquer une segmentation simple avant prédiction')
    parser.add_argument('--tta', action='store_true', help='Appliquer une TTA (flip horizontal) et moyennage des probabilités')
    parser.add_argument('--save', type=str, default=None, help='Chemin pour sauvegarder l\'image annotée au lieu d\'ouvrir une fenêtre')
    parser.add_argument('--threshold', type=float, default=0.5, help="Seuil de décision pour classer 'contaminée' (par défaut 0.5)")
    parser.add_argument('--gradcam', action='store_true', help='Générer une Grad-CAM overlay pour l\'image')
    parser.add_argument('--gradcam-layer', type=str, default=None, help='Nom de la couche conv à utiliser pour Grad-CAM (détecte automatiquement si absent)')
    parser.add_argument('--list-layers', action='store_true', help='Lister les couches du modèle et quitter (aide debug gradcam)')
    args = parser.parse_args()

    # Attach gradcam flags to function object for simple access
    predict_image._gradcam_request = args.gradcam
    predict_image._gradcam_layer = args.gradcam_layer

    if args.list_layers:
        # print a readable list of layers with types and output shapes
        def print_layers(m, prefix=''):
            for i, layer in enumerate(m.layers):
                try:
                    shape = getattr(layer, 'output_shape', None)
                except Exception:
                    shape = None
                print(f"{prefix}{i}: name={layer.name}, class={layer.__class__.__name__}, output_shape={shape}")
                # nested models
                if isinstance(layer, tf.keras.Model):
                    print_layers(layer, prefix=prefix + '  ')

        print('Model summary:')
        try:
            model.summary(print_fn=lambda s: print(s))
        except Exception:
            print('Unable to print model.summary()')
        print('\nLayer list:')
        print_layers(model)
        raise SystemExit(0)

    predict_image(args.image, debug=args.debug, do_segment=args.segment, do_tta=args.tta, save_path=args.save, threshold=args.threshold)

