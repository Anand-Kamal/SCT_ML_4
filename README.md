# Hand Gesture Recognition — Residual CNN

Classifying 10 hand gestures from infrared images using a small residual
convolutional neural network (CNN), trained on the
[LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)
dataset (Leap Motion infrared hand images, 10 subjects × 10 gestures).

## Overview

| | |
|---|---|
| **Task** | Multi-class image classification (10 hand gestures) |
| **Approach** | Deep learning — small residual CNN (not a pretrained model) |
| **Framework** | TensorFlow / Keras (Functional API) |
| **Dataset** | LeapGestRecog — 10 subjects, 10 gestures, ~20,000 grayscale frames |
| **Input** | 64×64 grayscale images |

## Why This Isn't a Naive Approach

Two choices in this project exist specifically to avoid a common mistake in
image classification projects — getting a misleadingly high accuracy that
won't hold up on real, unseen users:

- **Subject-disjoint splitting.** The 10 people in the dataset are split so
  that whole people are held out for validation and test — 6 for training, 2
  for validation, 2 for test — and no person appears in more than one set.
  A random split instead would let near-identical frames of the *same
  person* land in both train and test, since consecutive frames of one
  gesture look almost identical. That inflates accuracy to look close to
  100% without the model actually having learned to generalize to a new
  user. Section 10 of the notebook (`RUN_LEAKAGE_DEMO`) demonstrates this
  gap directly by training a second model on a random split for comparison.
- **Grayscale input.** The dataset is infrared, so color channels carry no
  real information — using grayscale cuts memory and training time with no
  accuracy cost.

## Model Architecture

A small residual CNN built with the Keras Functional API:

- Input: 64×64×1 grayscale image, rescaled to [0, 1] inside the model
- Data augmentation (random rotation, translation, zoom) — active only
  during training, built into the model graph so it can never be
  accidentally applied at inference time
- 3 residual stages (32 → 64 → 128 filters), each: two `Conv2D + BatchNorm +
  ReLU` blocks with a skip connection, followed by max pooling and spatial
  dropout
- Global average pooling → Dense(128, ReLU) → Dropout(0.4) → Dense(10, softmax)
- Loss: categorical cross-entropy with label smoothing (0.05)
- Optimizer: Adam (initial learning rate 1e-3)
- Callbacks: early stopping on validation loss (patience 6, restores best
  weights) and learning-rate reduction on plateau

## Methodology

1. **Download the dataset** via `kagglehub` (no manual download step needed).
2. **Index the files** into a table of `(path, label, subject)` before
   loading any images, so the subject-disjoint split logic stays simple.
3. **Preview one example per gesture** to sanity-check labels before
   processing.
4. **Load and resize** all images to 64×64 grayscale.
5. **Subject-disjoint split**: 6 train / 2 validation / 2 test subjects.
6. **Build `tf.data` pipelines** with shuffling, batching, and prefetching.
7. **Train** the residual CNN with early stopping and LR scheduling.
8. **Evaluate on the held-out test subjects** — this is the number that
   reflects real-world use on a brand-new person, not just held-out frames.
9. **Inspect results**: classification report, normalized confusion matrix,
   the model's most confident mistakes, and a random sample of predictions.
10. **(Optional) Leakage demo** — trains a second model on an ordinary
    random split to quantify how much accuracy is inflated by subject
    leakage. Off by default (`RUN_LEAKAGE_DEMO = True` to run).
11. **(Optional) Subject-wise cross-validation** — `GroupKFold` across 3
    folds, each holding out entire subjects, for a more robust accuracy
    estimate than a single split. Off by default (`RUN_GROUP_CV = True`,
    slowest option).
12. **Save the trained model** (`gesture_model.keras`) and provide a
    `predict_gesture()` helper for classifying a new image.

## Results

| Metric | Value |
|---|---|
| Test accuracy (unseen subjects) | 0.9585 |

Add a screenshot of the normalized confusion matrix and the sample
predictions grid here once you've run it, for example:


The dataset itself is not stored in the repo — it's downloaded automatically
via `kagglehub` when the notebook runs.

## Running the Notebook

1. Open [colab.research.google.com](https://colab.research.google.com) and
   upload `hand_gesture_recognition_cnn.ipynb`.
2. **Enable a GPU runtime**: Runtime → Change runtime type → GPU. Training a
   CNN on CPU will be far slower.
3. Run all cells top to bottom. `kagglehub` downloads the dataset
   automatically; no manual upload needed (a Kaggle account may still be
   required to accept the dataset's terms on first access).
4. Optionally set `RUN_LEAKAGE_DEMO = True` and/or `RUN_GROUP_CV = True` near
   the top of the notebook before running, for the extra validation
   experiments (both add meaningful runtime).

## Requirements

- Python 3.8+
- tensorflow
- opencv-python
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- kagglehub

These are preinstalled in Google Colab (aside from `kagglehub`, which Colab
installs automatically on import in recent runtimes, or via `pip install
kagglehub` if needed). Locally:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn kagglehub
```

## Notes & Limitations

- Trained from scratch rather than using a pretrained backbone — reasonable
  here since the images are simple, single-channel, and the dataset is
  large enough (~20,000 frames) for a small CNN to learn well, but a
  pretrained model (e.g. MobileNet) would be a natural next step for a
  harder or smaller dataset.
- The reported test accuracy reflects only the 2 held-out test subjects.
  Real-world performance on a genuinely new user's hand size, lighting, and
  camera angle may still differ from this dataset's infrared, controlled
  conditions.
- `predict_gesture()` expects a single image path and grayscale-loadable
  file (matching the dataset's format); it isn't set up for live webcam
  input out of the box.

## Acknowledgments

Dataset: [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog),
via Kaggle (`gti-upm/leapgestrecog`).
