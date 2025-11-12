"""
fine_tune.py

Script pour fine-tuner le modèle existant sur un dataset mixte (studio + web).
Usage example:
  python fine_tune.py --data-dir images --pretrained best_modele_detection_feuilles.h5 --unfreeze-last-n 4 --epochs 10

Le script :
- construit un ImageDataGenerator avec augmentations réalistes
- charge un modèle pré-entraîné si fourni
- fige les couches puis débloque les dernières N couches pour fine-tuning
- compile avec petit learning rate et entraîne
- sauvegarde le modèle final

"""
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight


def build_model(input_shape=(128, 128, 3)):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv2d'),
        MaxPooling2D(2, 2, name='max_pooling2d'),
        Conv2D(64, (3, 3), activation='relu', name='conv2d_1'),
        MaxPooling2D(2, 2, name='max_pooling2d_1'),
        Conv2D(128, (3, 3), activation='relu', name='conv2d_2'),
        MaxPooling2D(2, 2, name='max_pooling2d_2'),
        Flatten(name='flatten'),
        Dense(128, activation='relu', name='dense'),
        Dropout(0.5, name='dropout'),
        Dense(1, activation='sigmoid', name='dense_1')
    ])
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='images', help='Chemin vers le dossier contenant les sous-dossiers de classes')
    parser.add_argument('--pretrained', type=str, default=None, help='Chemin vers un modèle .h5 pré-entraîné à charger (optionnel)')
    parser.add_argument('--resume-from', type=str, default=None, help='Chemin vers un checkpoint .h5 pour reprendre l\'entraînement (load model and continue)')
    parser.add_argument('--unfreeze-last-n', type=int, default=3, help='Nombre de dernières couches à débloquer pour le fine-tuning')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', type=str, default='fine_tuned_modele.h5')
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--report', type=str, default=None, help='Chemin dossier pour enregistrer rapport (history.csv, model_summary.txt)')
    return parser.parse_args()


def main():
    args = parse_args()
    img_size = args.img_size

    # Augmentations plus réalistes
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.6, 1.4],
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=(img_size, img_size),
        batch_size=args.batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        args.data_dir,
        target_size=(img_size, img_size),
        batch_size=args.batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    # Charger / construire le modèle
    model = None
    if args.resume_from and os.path.exists(args.resume_from):
        print('Reprise depuis checkpoint :', args.resume_from)
        model = load_model(args.resume_from)
    elif args.pretrained and os.path.exists(args.pretrained):
        print('Chargement du modèle pré-entraîné :', args.pretrained)
        model = load_model(args.pretrained)
    else:
        print('Aucun modèle pré-entraîné fourni, construction d\'un modèle depuis zéro')
        model = build_model(input_shape=(img_size, img_size, 3))

    # Figer toutes les couches d'abord
    for layer in model.layers:
        layer.trainable = False

    # Débloquer les dernières N couches
    if args.unfreeze_last_n > 0:
        # parcourir par ordre et débloquer les dernières N layers
        to_unfreeze = args.unfreeze_last_n
        for layer in model.layers[-to_unfreeze:]:
            layer.trainable = True
        print(f"Débloqué les dernières {to_unfreeze} couches pour le fine-tuning")

    # Compile avec un LR bas
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    # callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(args.output, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7)
    ]

    # class weights
    y_train = train_gen.classes
    classes = np.unique(y_train)
    computed_weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, computed_weights)}
    print('Class weights:', class_weight_dict)

    steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
    validation_steps = max(1, val_gen.samples // val_gen.batch_size)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )

    print('Fine-tuning terminé. Modèle sauvegardé dans', args.output)
    # Save report if requested
    if args.report:
        os.makedirs(args.report, exist_ok=True)
        # save history as csv
        import csv
        hist_path = os.path.join(args.report, 'history.csv')
        with open(hist_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # header
            keys = list(history.history.keys())
            writer.writerow(['epoch'] + keys)
            for i in range(len(history.history[keys[0]])):
                row = [i+1] + [history.history[k][i] for k in keys]
                writer.writerow(row)
        # save model summary
        summary_path = os.path.join(args.report, 'model_summary.txt')
        try:
            with open(summary_path, 'w') as f:
                model.summary(print_fn=lambda s: f.write(s + '\n'))
        except Exception:
            with open(summary_path, 'w') as f:
                f.write('Unable to write model.summary()')
        print('Report saved to', args.report)


if __name__ == '__main__':
    main()
