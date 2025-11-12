Fine-tuning quick guide
=======================

This file explains how to prepare your images and run the fine-tuning script included in the repository.

1) Directory structure
----------------------

Prepare a folder called `images/` (or any name) with one subfolder per class. For your binary classifier the expected structure is:

images/
  ├─ Grape_Healthy/
  │    ├─ img001.jpg
  │    └─ img002.jpg
  └─ Grape_Leaf_Mildiou/
       ├─ imgA.jpg
       └─ imgB.jpg

The `ImageDataGenerator` expects subfolders named after the classes. Filenames can be arbitrary.

2) Recommended dataset mix
--------------------------

To reduce domain shift, include both:
- studio images (your existing dataset of isolated leaves), AND
- "in-the-field" images (web or taken on vine), both healthy and diseased.

Aim for at least ~100 images per class when possible. If you cannot collect many, consider synthetic augmentation (see below).

3) Running fine_tune.py
-----------------------

Basic run (build model from scratch):

```bash
python fine_tune.py --data-dir images --epochs 8 --batch-size 16 --output fine_tuned_model.h5
```

Resume from a pretrained model (recommended):

```bash
python fine_tune.py --data-dir images --pretrained best_modele_detection_feuilles.h5 --unfreeze-last-n 4 --epochs 8 --output fine_tuned_model.h5
```

Resume from a checkpoint (continue training):

```bash
python fine_tune.py --data-dir images --resume-from best_modele_detection_feuilles.h5 --unfreeze-last-n 4 --epochs 8 --output fine_tuned_model.h5
```

Save a report (CSV history + model summary):

```bash
python fine_tune.py --data-dir images --pretrained best_modele_detection_feuilles.h5 --unfreeze-last-n 4 --epochs 8 --report reports/run1
```

4) Options explained
---------------------

- `--data-dir`: path to the folder containing class subfolders.
- `--pretrained`: path to a .h5 model to load weights from before fine-tuning.
- `--resume-from`: path to a .h5 checkpoint to resume training from.
- `--unfreeze-last-n`: number of last layers to unfreeze and train (default 3).
- `--epochs`: number of epochs.
- `--batch-size`: batch size.
- `--output`: file where the best model is saved.
- `--report`: path to a folder where `history.csv` and `model_summary.txt` will be saved.

5) Tips
-------

- Monitor `history.csv` to check validation loss/accuracy and choose a threshold.
- If fine-tuning overfits quickly, reduce `unfreeze-last-n` or lower the learning rate.
- Consider adding more realistic augmentations or using the synthetic background collage approach if real web images are scarce.

6) Next steps
-------------

After fine-tuning, re-run `Prediction.py` on your web images (with `--gradcam --segment`) to verify that Grad-CAM now focuses on the leaf symptoms rather than the background.

If you want, I can also:
- add a script to synthesize images by pasting segmented leaves onto vine backgrounds,
- add automatic threshold calibration using a small validation set (ROC/precision/recall).

