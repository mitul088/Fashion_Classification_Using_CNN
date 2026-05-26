This repository serves as a portfolio compilation of computer vision project I developed locally between November 2024 and January 2025. It was migrated to GitHub in May 2026 for documentation and review purposes.

# Fashion Item Classification (Convolutional Neural Networks)

A beginner-friendly machine learning project that classifies grayscale fashion images into one of **10 categories** using **Convolutional Neural Networks (CNNs)** built with **TensorFlow/Keras**. The full walkthrough lives in the Jupyter notebook [`Fashion Item Classification (Convolutional Neural Networks).ipynb`](Fashion%20Item%20Classification%20(Convolutional%20Neural%20Networks).ipynb).

---

## What this project does

Given a 28×28 pixel image of a clothing item or accessory, the model predicts which class it belongs to—for example, a sneaker, a dress, or a bag. The notebook covers the full pipeline:

1. Load and explore the Fashion-MNIST data
2. Visualize sample images
3. Preprocess pixels and reshape data for a CNN
4. Train a baseline CNN
5. Evaluate with accuracy, confusion matrix, and classification report
6. Train an improved CNN (more filters + dropout) and compare results

**Reported test accuracy (from notebook runs):**

| Model | Test accuracy |
|-------|----------------|
| Baseline CNN (32 filters) | ~91.0% |
| Improved CNN (64 filters + dropout) | ~92.8% |

Actual results may vary slightly depending on hardware, TensorFlow version, and random seeds.

---

## Dataset

This project uses the **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)** dataset: 70,000 grayscale images of fashion products.

| Split | Samples | Shape per row (CSV) |
|-------|---------|---------------------|
| Training | 60,000 | 785 columns (`label` + 784 pixels) |
| Test | 10,000 | 785 columns |

- **Image size:** 28 × 28 pixels (784 values per image)
- **Pixel range:** 0 (light) to 255 (dark), stored as integers in the CSV files

### Class labels

| Label | Item |
|-------|------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### Data files (included)

```
input/
├── fashion-mnist_train.csv
└── fashion-mnist_test.csv
```

Each CSV has a `label` column plus `pixel1` … `pixel784`. The notebook loads them with pandas:

```python
fashion_train_df = pd.read_csv('input/fashion-mnist_train.csv')
fashion_test_df = pd.read_csv('input/fashion-mnist_test.csv')
```

---

## Project structure

```
Fashion item classification/
├── README.md
├── Fashion Item Classification (Convolutional Neural Networks).ipynb
└── input/
    ├── fashion-mnist_train.csv
    └── fashion-mnist_test.csv
```

---

## Requirements

- **Python** 3.8+ recommended
- **Jupyter** (Notebook or JupyterLab) to run the `.ipynb` file

### Python packages

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `pandas` | Load CSV data |
| `matplotlib` | Plot images and grids |
| `seaborn` | Confusion matrix heatmap |
| `scikit-learn` | Train/validation split, metrics |
| `tensorflow` | Build and train the CNN |

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

> **Note:** Training uses CPU or GPU depending on your TensorFlow install. GPU speeds up training but is not required.

---

## How to run

1. **Clone or download** this folder to your machine.

2. **Open a terminal** in the project directory:

   ```bash
   cd "Fashion item classification"
   ```

3. **Start Jupyter** and open the notebook:

   ```bash
   jupyter notebook
   ```

   Open `Fashion Item Classification (Convolutional Neural Networks).ipynb`.

4. **Run cells in order** from top to bottom. Early cells only need the CSV files under `input/`. Training (Steps 4–6) can take a while because each model trains for **50 epochs** with `batch_size=512`.

5. **Kernel:** Use a Python environment where TensorFlow is installed. Restart the kernel if you change packages.

---

## Notebook workflow (summary)

### Step 1 — Problem statement

Defines the 10-class image classification task and Fashion-MNIST properties.

### Step 2 — Data preprocessing

- Import libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `random`)
- Read train/test CSVs into NumPy arrays

### Step 3 — Visualizing the dataset

- Confirm shapes: training `(60000, 785)`, testing `(10000, 785)`
- Display random 28×28 images and a 15×15 grid of samples with labels

### Step 4 — Training the baseline model

**Preprocessing:**

- Features: all columns except `label`, scaled to `[0, 1]` by dividing by 255
- Labels: first column (`label`)
- 80/20 split of training data into train / validation (`train_test_split`, `test_size=0.2`)
- Reshape to `(samples, 28, 28, 1)` for Conv2D input

**Baseline CNN architecture:**

```
Conv2D (32 filters, 3×3, ReLU) → MaxPool2D (2×2)
→ Conv2D (32 filters, 3×3, ReLU) → MaxPool2D (2×2)
→ Flatten
→ Dense (128, ReLU)
→ Dense (10, sigmoid)   # output layer
```

- **Optimizer:** Adam  
- **Loss:** `sparse_categorical_crossentropy`  
- **Metrics:** accuracy  
- **Training:** 50 epochs, batch size 512, validation on held-out 20% of training data  

### Step 5 — Evaluating the baseline model

- Test set accuracy
- Sample prediction grid (predicted vs true label)
- Confusion matrix (seaborn heatmap)
- Classification report (precision, recall, F1 per class)

### Step 6 — Improving the model

Same pipeline with a deeper regularized network:

- **64 filters** (instead of 32) in both conv blocks
- **Dropout (0.25)** after pooling and after the dense layer

Then repeat evaluation (accuracy, plots, confusion matrix, classification report).

---

## Model design notes

- **Input shape:** `(28, 28, 1)` — single-channel grayscale.
- **Why CNNs?** Convolutional layers learn spatial patterns (edges, textures) that work well on images, unlike a plain dense network on flattened pixels alone.
- **Validation split:** 20% of training data is used during `fit()` to monitor `val_accuracy` and reduce overfitting awareness; the final **test** set (`fashion-mnist_test.csv`) is only used in Step 5/6 evaluation.
- **Output activation:** The notebook uses `sigmoid` on the last layer with `sparse_categorical_crossentropy`. In many projects, a **softmax** output is paired with this loss for multi-class classification; the notebook’s setup still trains and evaluates as documented above.

---

## Tips for readers

- **Run all cells in order** the first time; later you can re-run only training/evaluation cells after architecture changes.
- If `predict_classes` raises an error on newer TensorFlow versions, use:

  ```python
  import numpy as np
  predict_classes = np.argmax(cnn.predict(X_test), axis=1)
  ```

- Training for 50 epochs is intentional in the notebook; you can lower `epochs` for a quicker experiment.
- Confusing classes (e.g. shirt vs T-shirt vs pullover) often show up in the confusion matrix—that is expected on Fashion-MNIST.

---

## Credits

- Dataset: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando Research)
- Built as an educational CNN classification tutorial in Jupyter

---

## License

If you share or adapt this project, follow the license terms of the Fashion-MNIST dataset and any course or repository requirements that apply to your copy of the materials.
