# Capstone Project: Satellite Image Classification for Fertilizer Sales Forecasting
### Agricultural vs. Non-Agricultural: one CNN, built in both Keras and PyTorch

> **About this repository:** a capstone project on **geospatial AI and binary image classification**. **Part 1** covers the data-handling process (loading methods, efficient pipelines, data augmentation). **Part 2** builds, tunes, and evaluates two independent CNNs, one in Keras and one in PyTorch, then compares their results and workflows. It is written both as a **learning reference** (to memorize the key ideas) and as a **guide for others** to follow.

## Table of Contents

- **[Project at a Glance](#project-at-a-glance)**
- **[Why This Project?](#why-this-project)**
- **[Project Goals](#project-goals)**
- **[How the Project Is Organized](#how-the-project-is-organized)**
- **[Conventions Used in Both Frameworks](#conventions-used-in-both-frameworks)**
- **[Part 1: Data Handling](#part-1-data-handling)**
  - [1.1 Data Loading Methods](#11-data-loading-methods)
  - [1.2 Building Efficient Data Pipelines](#12-building-efficient-data-pipelines)
  - [1.3 Data Augmentation in PyTorch](#13-data-augmentation-in-pytorch)
  - [1.4 Part 1 Summary](#14-part-1-summary)
- **[Part 2: CNN Classifier, Keras vs. PyTorch](#part-2-cnn-classifier-keras-vs-pytorch)**
  - [2.1 Problem Framing](#21-problem-framing)
  - [2.2 Designing a Fair Comparison](#22-designing-a-fair-comparison)
  - [2.3 Hyperparameters](#23-hyperparameters)
  - [2.4 Data Loading: Keras vs. PyTorch](#24-data-loading-keras-vs-pytorch)
  - [2.5 Data Augmentation: Keras vs. PyTorch](#25-data-augmentation-keras-vs-pytorch)
  - [2.6 Model Architecture: Keras vs. PyTorch](#26-model-architecture-keras-vs-pytorch)
  - [2.7 Training Workflow: Keras vs. PyTorch](#27-training-workflow-keras-vs-pytorch)
  - [2.8 Evaluating the Classifier](#28-evaluating-the-classifier)
  - [2.9 Comparing Keras and PyTorch](#29-comparing-keras-and-pytorch)
  - [2.10 Conclusions](#210-conclusions)
- **[Appendix A: Data and Setup](#appendix-a-data-and-setup)**
- **[Appendix B: Reproducibility Checklist](#appendix-b-reproducibility-checklist)**
- **[Appendix C: Suggested Presentation Outline](#appendix-c-suggested-presentation-outline)**

---

## Project at a Glance

| Item | Description |
|---|---|
| **Problem** | Binary image classification: **agricultural** vs. **non-agricultural** satellite images |
| **Business use** | Estimate agricultural land to support **fertilizer sales forecasting** |
| **Data** | `[Dataset name, source, image size, number of images, class balance, license: to fill in]` |
| **Models** | Two independent CNNs with equivalent architectures: one in **Keras**, one in **PyTorch** |
| **Skills practiced** | Data pipelines, data augmentation, CNNs, hyperparameter tuning, evaluation (precision, recall, F1, ROC-AUC), framework comparison, geospatial AI |
| **Deliverables** | Notebooks, reusable code in `src/`, saved results (metrics and figures), and this README |

## Why This Project?

**The business problem.** Fertilizer sellers need to know where agricultural land is, and how much of it, to forecast demand. Satellite imagery covers large regions cheaply, but it is far too large to label by hand. A CNN that labels each image tile as agricultural or non-agricultural turns raw imagery into an estimate of farmland that a sales-forecasting model can use.

> **Scope:** the forecasting model itself is outside this repository. This project delivers the **classifier** that supplies its input.

**Why precision comes first.** Which error is worse depends on the use case:

| Error | What happens | Effect on the forecast |
|---|---|---|
| **False positive** (non-agricultural tile labeled agricultural) | Farmland is over-counted | Fertilizer demand is **over-forecast** |
| **False negative** (agricultural tile labeled non-agricultural) | Farmland is missed | Fertilizer demand is **under-forecast** |

We report **precision** first, and always alongside recall, F1, and ROC-AUC, so the trade-off stays visible.

## Project Goals

| # | Goal | Why it matters |
|---|---|---|
| 1 | **Validate the models** | Two independent implementations should reach comparable results. If they disagree, one of them probably has a bug, so each model cross-checks the other. |
| 2 | **Understand the distinct workflows** | Knowing how Keras and PyTorch differ (data loading, augmentation, training loop, evaluation) gives us wider career flexibility. |

## How the Project Is Organized

| Part | Topic | Question it answers |
|---|---|---|
| **Part 1** | Data handling | How do we load, batch, and augment images efficiently? |
| **Part 2** | CNN classifier: Keras vs. PyTorch | How do we build, tune, evaluate, and fairly compare two CNNs? |

```text
satellite-agri-classifier/
├── README.md
├── requirements.txt
├── data/                       # NOT committed; see "Data" section for download steps
│   └── splits/                 # train.csv, val.csv, test.csv (path, label)
├── notebooks/
│   ├── 01_data_loading_and_pipelines.ipynb
│   ├── 02_data_augmentation_pytorch.ipynb
│   ├── 03_cnn_keras.ipynb
│   ├── 04_cnn_pytorch.ipynb
│   └── 05_evaluation_and_comparison.ipynb
├── src/
│   ├── data_pipeline.py        # generators, Dataset classes, tf.data helpers
│   ├── models_keras.py
│   ├── models_pytorch.py
│   └── evaluate.py             # ONE shared evaluation script for both frameworks
└── results/
    ├── keras/                  # predictions.csv, history, best model
    ├── pytorch/
    └── figures/                # confusion matrices, ROC curves, augmentation samples
```

## Conventions Used in Both Frameworks

These choices are fixed once and used everywhere, so the two models can be compared fairly.

| Convention | Value |
|---|---|
| **Positive class** | `agricultural` = **1**, `non_agricultural` = 0 |
| **Splits** | Created **once** (stratified, fixed seed), stored as `train.csv`, `val.csv`, `test.csv`, and reused by both frameworks |
| **Image size** | `[H × W × 3]`, set once (e.g., 64×64) |
| **Pixel scale** | Float values in `[0, 1]` |
| **Decision threshold** | 0.5 (also analyzed at other thresholds in Part 2) |
| **Random seed** | 42 in NumPy, Python, Keras, and PyTorch |

---

## Part 1: Data Handling

Good models start with good data pipelines. This part covers **how images get from disk into the model** (loading), **how to build the pipeline efficiently** (custom generators vs. built-in utilities), and **how to make the training data more varied** (data augmentation in PyTorch).

```text
Files on disk → decode → resize → augment (training only) → batch → prefetch → model
```

### 1.1 Data Loading Methods

There are two basic ways to get images into a model.

#### Bulk loading (memory-based)

**Idea:** read **all** images into one big array in RAM, then slice batches from it.

```python
import numpy as np
from PIL import Image

def load_all(paths, size=(64, 64)):
    # uint8 array of shape (N, H, W, 3), held entirely in memory
    return np.stack([np.asarray(Image.open(p).convert("RGB").resize(size)) for p in paths])

X_train = load_all(train_paths)
model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(X_val, y_val))
```

#### Sequential loading (generator-based)

**Idea:** read images **batch by batch, on demand**. Only the current batch lives in memory.

```python
import numpy as np
from PIL import Image

def batch_generator(paths, labels, batch_size=32, size=(64, 64), shuffle=True):
    idx = np.arange(len(paths))
    while True:                                   # endless: Keras stops after steps_per_epoch
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            batch = idx[start:start + batch_size]
            x = np.stack([np.asarray(Image.open(paths[i]).convert("RGB").resize(size))
                          for i in batch]).astype("float32") / 255.0
            yield x, labels[batch]

steps = int(np.ceil(len(train_paths) / 32))
model.fit(batch_generator(train_paths, y_train), steps_per_epoch=steps, epochs=30)
```

> **Note:** this stand-alone example divides by 255 itself. If the model already contains a `Rescaling` layer (as in Section 2.6), skip that division so the pixels are not scaled twice.

#### Memory-based vs. generator-based

| Aspect | Bulk (memory-based) | Sequential (generator-based) |
|---|---|---|
| **Memory use** | Whole dataset in RAM | One batch (plus a small buffer) |
| **Startup time** | Slow (everything is read first) | Fast (starts immediately) |
| **Speed per epoch** | Fast (no disk reads after loading) | Slower, unless loading overlaps with training |
| **Shuffling** | Easy and global | Needs care (shuffle the index list each epoch) |
| **Augmentation** | Applied to arrays, or pre-computed copies | Applied on the fly, a new random version every epoch |
| **Scales to large data** | No | Yes |
| **Code complexity** | Very simple | A little more code |
| **Best for** | Small datasets that fit comfortably in RAM | Large datasets, or when augmenting on the fly |

**How much memory does bulk loading need?**

$$\text{bytes} = N \times H \times W \times C \times \text{bytes per value}$$

| Example (illustrative sizes) | `uint8` (1 byte) | `float32` (4 bytes) |
|---|---:|---:|
| 6,000 images of 64×64×3 | ≈ 74 MB | ≈ 295 MB |
| 100,000 images of 256×256×3 | ≈ 19.7 GB | ≈ 78.6 GB |

> **Tip:** keep images as `uint8` in memory and convert to `float32` one batch at a time. Replace the example sizes with our own dataset.

### 1.2 Building Efficient Data Pipelines

An efficient pipeline keeps the GPU busy: while the model trains on batch *k*, the next batches are already being loaded and prepared. We build and compare two approaches.

| | **Approach A: custom data generators** | **Approach B: built-in utilities** |
|---|---|---|
| **Keras** | `keras.utils.PyDataset` (or a Python generator) | `keras.utils.image_dataset_from_directory` + `tf.data` |
| **PyTorch** | `torch.utils.data.Dataset` + `DataLoader` | `torchvision.datasets.ImageFolder` + `DataLoader` |

#### Approach A: custom data generators

**Keras: a class-based generator (`PyDataset`).** It knows its own length and is safe for parallel loading, which a plain `yield` generator is not.

```python
import numpy as np
import keras
from PIL import Image

class SatelliteSequence(keras.utils.PyDataset):
    def __init__(self, paths, labels, batch_size=32, image_size=(64, 64), shuffle=True, **kwargs):
        super().__init__(**kwargs)               # kwargs: workers, use_multiprocessing, max_queue_size
        self.paths, self.labels = np.array(paths), np.array(labels)
        self.batch_size, self.image_size, self.shuffle = batch_size, image_size, shuffle
        self.on_epoch_end()

    def __len__(self):                           # number of batches per epoch
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):                  # one batch, loaded from disk on demand
        batch = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = np.stack([np.asarray(Image.open(self.paths[i]).convert("RGB").resize(self.image_size))
                      for i in batch]).astype("float32")
        return x, self.labels[batch].astype("float32")

    def on_epoch_end(self):                      # reshuffle after every epoch
        self.indices = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
```

**PyTorch: a custom `Dataset` and a `DataLoader`.** This is also the pattern used for augmentation in Section 1.3.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SatelliteDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)

train_loader = DataLoader(SatelliteDataset(train_paths, y_train, train_tf),
                          batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
```

#### Approach B: built-in utilities

**Keras: `image_dataset_from_directory`.** It reads images from folders, resizes them, batches them, and returns a `tf.data.Dataset`.

```text
data/train/agricultural/*.jpg
data/train/non_agricultural/*.jpg      (same layout for data/val and data/test)
```

```python
import keras
import tensorflow as tf

common = dict(
    labels="inferred",
    label_mode="binary",                                  # float labels 0./1. for a sigmoid output
    class_names=["non_agricultural", "agricultural"],     # fixes the order: agricultural = 1
    image_size=(64, 64),
    batch_size=32,
)
train_ds = keras.utils.image_dataset_from_directory("data/train", shuffle=True, seed=42, **common)
val_ds   = keras.utils.image_dataset_from_directory("data/val",   shuffle=False, **common)
test_ds  = keras.utils.image_dataset_from_directory("data/test",  shuffle=False, **common)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)            # overlap loading with training
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)
```

> **Why `class_names` matters:** folders are otherwise sorted alphabetically, which would make `agricultural` = 0. Setting the order explicitly keeps **agricultural = 1 (the positive class)**, so precision and recall mean the same thing in both frameworks.

> **Caching caveat:** `.cache()` speeds up small datasets, but on a batched dataset it freezes the batch composition after the first epoch. Use it only when the data fits in memory, and put random augmentation *after* the cache (for example, as layers inside the model).

> **Deprecated:** the older `ImageDataGenerator` is deprecated in Keras. Prefer `image_dataset_from_directory` plus preprocessing layers for new code.

**PyTorch: `ImageFolder` + `DataLoader`.** Same folder layout, one line to load:

```python
from torchvision import datasets
train_ds = datasets.ImageFolder("data/train", transform=train_tf)
print(train_ds.class_to_idx)      # sorted alphabetically: check which class is 1!
```

> **Watch out:** `ImageFolder` also sorts class folders alphabetically, so `agricultural` would be 0. Either remap the labels (e.g., `target_transform=lambda y: 1 - y`, and cast them to float for the loss) or use the custom `Dataset` above with an explicit label mapping.

#### Efficiency settings side by side

| Concern | Keras / `tf.data` | PyTorch |
|---|---|---|
| **Parallel decoding** | Handled by the utility (`num_parallel_calls=tf.data.AUTOTUNE` in custom `map` calls) | `DataLoader(num_workers=N)` |
| **Overlap loading with training** | `.prefetch(tf.data.AUTOTUNE)` | Worker processes prefetch batches (`prefetch_factor`) |
| **Caching** | `.cache()` | No built-in cache; cache decoded images inside the `Dataset` |
| **Faster transfer to the GPU** | Automatic | `pin_memory=True` |
| **Shuffling** | `shuffle=True` | `shuffle=True` in `DataLoader` |
| **Multiprocessing for custom loaders** | `PyDataset(workers=..., use_multiprocessing=True)` | `num_workers > 0` |

#### Custom generators vs. built-in utilities

| Criterion | Custom generator / `Dataset` | Built-in utility |
|---|---|---|
| **Control** | Full: any file format, any logic | Limited to the supported layout |
| **Code to write** | More | Very little |
| **Speed** | As good as our implementation | Usually well optimized |
| **Reusing a manifest (`train.csv`)** | Natural | Needs the folder layout |
| **Best for** | Custom splits, extra metadata, unusual formats | Standard folder-per-class image data |

#### Pipeline checklist

- [ ] **Split first, augment later:** create train/val/test splits before any augmentation
- [ ] **Stratify** the split so class balance is the same in each set
- [ ] **Never augment** validation or test data
- [ ] **Compute normalization statistics on the training set only** (if we use mean/std)
- [ ] Set **`shuffle=True` for training only**; keep `shuffle=False` for val/test so predictions line up with labels
- [ ] Use the **same split files** in both frameworks

### 1.3 Data Augmentation in PyTorch

**What it is.** Data augmentation creates new, slightly different versions of training images (flipped, rotated, re-lit) on the fly. The label stays the same, but the model sees more variety, which **reduces overfitting** and improves generalization.

| | **Online augmentation** | **Offline augmentation** |
|---|---|---|
| **When** | Every time an image is loaded | Once, before training |
| **Storage** | None | Augmented copies saved to disk |
| **Variety** | A new random version every epoch | A fixed set of versions |
| **Used here** | Yes (in `Dataset.__getitem__`) | No |

#### The transforms

```python
import torch
from torchvision.transforms import v2

train_tf = v2.Compose([
    v2.ToImage(),
    v2.Resize((64, 64)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.ColorJitter(contrast=0.1),
    v2.ToDtype(torch.float32, scale=True),     # uint8 [0, 255] → float32 [0, 1]
])

eval_tf = v2.Compose([                          # NO random transforms for validation and test
    v2.ToImage(),
    v2.Resize((64, 64)),
    v2.ToDtype(torch.float32, scale=True),
])
```

`train_tf` goes into the training `Dataset`; `eval_tf` goes into the validation and test datasets.

#### Which transforms suit satellite images?

| Transform | What it simulates | Suitable for satellite tiles? |
|---|---|---|
| `RandomHorizontalFlip`, `RandomVerticalFlip` | Different orientation | **Yes.** Overhead imagery usually has no canonical "up" |
| `RandomRotation` | Different orientation or sensor angle | **Yes.** Small angles leave empty corners (filled black by default) |
| `ColorJitter` | Lighting, season, atmosphere | **Use mildly.** Color is a key cue for vegetation, so strong changes (especially hue) could change the true label |
| `RandomResizedCrop` | Different scale and framing | **Carefully.** A crop may remove the evidence for the label |
| `GaussianBlur`, noise | Haze, sensor noise | Optional |
| `Normalize` | Not augmentation: standardizes the inputs | Apply **identically** to train, val, and test |

#### Checking and reproducing augmentation

```python
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open(sample_path).convert("RGB")
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for ax in axes.ravel():
    ax.imshow(train_tf(img).permute(1, 2, 0))      # (C, H, W) → (H, W, C)
    ax.axis("off")
plt.savefig("results/figures/augmentation_examples.png", bbox_inches="tight")
```

- **Always look at augmented samples** before training. Wrong settings (too much rotation or color shift) are easy to spot visually.
- **Seed everything** (`torch.manual_seed(42)`) and give `DataLoader` workers a `worker_init_fn` or a seeded `generator`, so augmentation is reproducible.

> **Connection to Part 2:** Section 2.5 compares this PyTorch approach with Keras preprocessing layers, using equivalent settings.

### 1.4 Part 1 Summary

| Topic | Key takeaway |
|---|---|
| **Bulk loading** | Simple and fast, but memory grows with the dataset |
| **Sequential (generator) loading** | Constant memory, scales to large data, ideal for on-the-fly augmentation |
| **Custom generators** | Maximum control (`PyDataset`, `Dataset` + `DataLoader`) |
| **Built-in utilities** | Least code (`image_dataset_from_directory`, `ImageFolder`) |
| **Augmentation in PyTorch** | `torchvision.transforms.v2` in the training `Dataset` only; choose transforms that keep the label true |

---

## Part 2: CNN Classifier, Keras vs. PyTorch

In this part we build the **same CNN twice**, once in Keras and once in PyTorch, tune its hyperparameters, evaluate it with the same metrics, and compare the two workflows.

### 2.1 Problem Framing

| Item | Choice |
|---|---|
| **Task** | Binary classification of an image tile: agricultural (1) or non-agricultural (0) |
| **Input** | RGB satellite image tile, `[H × W × 3]` |
| **Output** | One number per image: the probability of "agricultural" |
| **Loss** | Binary cross-entropy |
| **Decision rule** | Predict agricultural if probability ≥ threshold (0.5 by default) |

| | **Keras** | **PyTorch** |
|---|---|---|
| **Last layer** | `Dense(1, activation="sigmoid")` | `Linear(128, 1)` (raw logit) |
| **Loss** | `"binary_crossentropy"` | `nn.BCEWithLogitsLoss()` (applies the sigmoid internally) |
| **Probability at prediction time** | The model output | `torch.sigmoid(logit)` |

The two setups are mathematically equivalent: PyTorch simply folds the sigmoid into the loss for numerical stability.

### 2.2 Designing a Fair Comparison

A comparison is only meaningful if the **only thing that differs is the framework**. We therefore fix the following settings in both.

#### Shared configuration (proposed starting point; edit to match the final runs)

| Setting | Value (identical in Keras and PyTorch) |
|---|---|
| **Data and splits** | Same train/val/test files (from `data/splits/`) |
| **Image size and scaling** | `[H × W × 3]`, pixel values in `[0, 1]` |
| **Augmentation** | Horizontal and vertical flips, rotation ±15°, contrast ±10% |
| **Architecture** | Same layers and sizes (**109,889** trainable parameters, see Section 2.6) |
| **Loss** | Binary cross-entropy |
| **Optimizer** | Adam |
| **Learning rate** | 1e-3 |
| **Batch size** | 32 |
| **Epochs** | Maximum 30, with the same early-stopping rule (validation loss, patience 5, restore best weights) |
| **Decision threshold** | 0.5 |
| **Seeds** | 42, 43, 44 (three runs per framework) |
| **Hardware** | Same machine and GPU |
| **Evaluation code** | One shared script (`src/evaluate.py`) applied to saved predictions |

#### Known differences that can still bias the results

Even with the same settings, the two frameworks have different defaults. We set each of these **explicitly**:

| Difference | Keras default | PyTorch default | What we do |
|---|---|---|---|
| **Initial weights** | Glorot uniform, zero biases | Kaiming-uniform variant, uniform biases | Use **He normal** in both (Section 2.6) |
| **Adam epsilon** | 1e-7 | 1e-8 | Set the same value in both |
| **Convolution padding** | `"valid"` | `0` | Set explicitly (`"same"` / `padding=1`) |
| **Rotation fill** | Reflect | Black (0) | Set Keras `fill_mode="constant"` |
| **Resize** | Bilinear, no antialiasing | Bilinear, antialiasing on for tensors | **Pre-resize images once** and save them, so neither framework resizes during training |
| **L2 / weight decay** | `l2(λ)` adds λ·Σw² to the loss (gradient 2λw) | `weight_decay=λ′` adds λ′·w to the gradient | Use `λ′ = 2λ`, or use `AdamW` in both with the same `weight_decay` (the defaults differ: 0.004 vs. 0.01) |
| **Batch norm momentum** (if added) | 0.99 | 0.1 (equals 0.9 in Keras terms) | Set equivalent values |
| **Data layout** | Channels last (NHWC) | Channels first (NCHW) | No effect on results, but remember it when comparing shapes |
| **Random number generators** | Different | Different | Runs are never bit-identical, so **repeat with several seeds** |

#### Run protocol

1. Train each framework with **3 seeds** and report **mean ± standard deviation**.
2. Save every model's **test-set probabilities** to a CSV (`results/<framework>/predictions.csv`).
3. Compute **all metrics with the same script** on those files, never with each framework's built-in metrics.
4. Record **training time per epoch**, the **epoch of the best validation loss**, and the **parameter count**.

> **How to read the results:** if the difference between Keras and PyTorch is smaller than the seed-to-seed variation within one framework, we cannot claim that one framework is better.

### 2.3 Hyperparameters

Hyperparameters are the settings we choose **before** training. They fall into three groups.

| Group | Hyperparameters | What they control |
|---|---|---|
| **Optimization** | Optimizer, learning rate, momentum, learning-rate schedule | How the weights are updated |
| **Regularization** | Dropout, weight decay, augmentation, early stopping, batch normalization | How strongly we prevent overfitting |
| **Training budget** | Batch size, number of epochs | How much and how often the model learns |

#### 2.3.1 Optimization hyperparameters

| Hyperparameter | What it controls | Common starting point | Keras | PyTorch |
|---|---|---|---|---|
| **Optimizer** | The update rule (SGD, Adam, AdamW, RMSprop) | Adam | `keras.optimizers.Adam` | `torch.optim.Adam` |
| **Learning rate** | Step size. Too high: unstable or diverging. Too low: slow training | 1e-3 for Adam | `learning_rate=1e-3` | `lr=1e-3` |
| **Momentum** (SGD only) | How much past gradients are carried forward | 0.9 | `SGD(momentum=0.9)` | `SGD(momentum=0.9)` |
| **LR schedule** | Lowers the learning rate when progress stalls | Halve after 2 epochs without improvement | `ReduceLROnPlateau` callback | `lr_scheduler.ReduceLROnPlateau` |

#### 2.3.2 Regularization hyperparameters

| Hyperparameter | What it does | Common starting point | Keras | PyTorch |
|---|---|---|---|---|
| **Dropout rate** | Randomly switches off a fraction of neurons during training | 0.3 to 0.5 before the output layer | `Dropout(0.5)` | `nn.Dropout(0.5)` |
| **Weight decay (L2)** | Penalizes large weights | 1e-4 | `kernel_regularizer=l2(1e-4)` | `weight_decay=2e-4` (see the table in 2.2) |
| **Data augmentation** | Adds variety to the training images | Flips, small rotation, mild contrast | Preprocessing layers | `transforms.v2` |
| **Early stopping** | Stops when validation loss stops improving | Patience 5 | `EarlyStopping` | Manual loop |
| **Batch normalization** | Stabilizes training, mild regularizer | Optional experiment | `BatchNormalization()` | `nn.BatchNorm2d` |

In both frameworks, the dropout rate is the probability of **dropping** a neuron.

#### 2.3.3 Batch size and epoch count

| Hyperparameter | Effect when **smaller** | Effect when **larger** |
|---|---|---|
| **Batch size** | Noisier gradients (can regularize), more updates per epoch, slower per epoch | Smoother gradients, faster per epoch, more GPU memory; may need a higher learning rate |
| **Epoch count** | Underfitting: the model has not converged yet | Overfitting: validation loss rises while training loss keeps falling |

$$\text{updates per epoch} = \left\lceil \frac{N_{\text{train}}}{\text{batch size}} \right\rceil$$

We set a **maximum number of epochs** and let **early stopping** choose the actual number, with the same rule in both frameworks. We then report the number of epochs actually run.

#### 2.3.4 Tuning strategy

1. **Tune on the validation set only.** The test set is used **once**, at the end.
2. Use the **same search space and the same budget** for both frameworks.
3. Start with the learning rate and batch size, then dropout and weight decay.
4. Pick the best **common** configuration for the final comparison, so both frameworks run with identical settings. Optionally add an "individually tuned" result as an extra row.

**Example search space** (small enough to run in both frameworks):

| Hyperparameter | Values |
|---|---|
| Learning rate | 1e-2, 1e-3, 1e-4 |
| Batch size | 16, 32, 64 |
| Dropout | 0.3, 0.5 |

**Tuning log template:**

| Run | Framework | Optimizer | LR | Batch size | Dropout | Weight decay | Epochs run | Val loss | Val precision | Val F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Keras | Adam | 1e-3 | 32 | 0.5 | 0 | TBD | TBD | TBD | TBD |
| 2 | PyTorch | Adam | 1e-3 | 32 | 0.5 | 0 | TBD | TBD | TBD | TBD |

---

### 2.4 Data Loading: Keras vs. PyTorch

Section 1.2 explained both approaches in detail. Here we compare what each framework does for the CNN experiments.

| Step | Keras | PyTorch |
|---|---|---|
| **Read files and labels** | `image_dataset_from_directory` (built-in) or `PyDataset` (custom) | `ImageFolder` (built-in) or custom `Dataset` |
| **Batching** | `batch_size=` argument | `DataLoader(batch_size=...)` |
| **Shuffling** | `shuffle=True` (training only) | `shuffle=True` (training only) |
| **Parallel loading** | `.prefetch(tf.data.AUTOTUNE)` | `num_workers=N`, `pin_memory=True` |
| **Output object** | `tf.data.Dataset` of `(images, labels)` | `DataLoader` of `(images, labels)` |
| **Label shape** | `(batch, 1)` float | `(batch,)` float |
| **Pixel values** | 0 to 255 (scaled inside the model by `Rescaling`) | 0 to 1 (`ToDtype(..., scale=True)`) |

**Identical splits in both frameworks.** The built-in Keras utility needs a folder layout, so we create the `train/val/test` folders **once** from the shared split files, and use the same folders for PyTorch's `ImageFolder`, or the same CSVs for the custom `Dataset`:

```python
import shutil
import pandas as pd
from pathlib import Path

for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/splits/{split}.csv")            # columns: path, label
    for path, label in zip(df["path"], df["label"]):
        cls = "agricultural" if label == 1 else "non_agricultural"
        dest = Path("data") / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dest / Path(path).name)
```

### 2.5 Data Augmentation: Keras vs. PyTorch

| | **Keras** | **PyTorch** |
|---|---|---|
| **Where it lives** | **Preprocessing layers** inside the model | **Transforms** inside the `Dataset` |
| **Where it runs** | With the model (on the GPU when one is used) | In `DataLoader` worker processes (CPU) |
| **Active when** | Only during training (automatically) | Only where we pass `train_tf` (we choose) |
| **Risk** | Few: the framework handles training vs. inference | Forgetting to use `eval_tf` for validation and test |

**Equivalent settings** (as used in the shared configuration):

| Augmentation | Keras layer | PyTorch transform (`v2`) |
|---|---|---|
| Flip horizontally and vertically | `RandomFlip("horizontal_and_vertical")` | `RandomHorizontalFlip(0.5)` + `RandomVerticalFlip(0.5)` |
| Rotation ±15° | `RandomRotation(15 / 360, fill_mode="constant")` | `RandomRotation(15)` |
| Contrast ±10% | `RandomContrast(0.1)` | `ColorJitter(contrast=0.1)` |
| Scale pixels to `[0, 1]` | `Rescaling(1 / 255)` | `ToDtype(torch.float32, scale=True)` |

```python
import keras
from keras import layers

augment = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(15 / 360, fill_mode="constant"),   # factor is a fraction of 360°
    layers.RandomContrast(0.1),
])
```

> **Careful with units:** Keras `RandomRotation` takes a **fraction of a full turn** (`15 / 360` ≈ ±15°), while PyTorch `RandomRotation` takes **degrees**.

> **Equivalent, not identical:** the random operations are implemented differently in the two libraries, so exact pixel values will differ. Averaging over several seeds (Section 2.2) absorbs this.

### 2.6 Model Architecture: Keras vs. PyTorch

Both models use the **same architecture**: three convolution blocks, global average pooling, and a small classifier head.

| # | Layer | Output shape (for a 64×64×3 input) | Parameters |
|---:|---|---|---:|
| 1 | Conv 3×3, 32 filters, ReLU, `same` padding | 64×64×32 | 896 |
| 2 | Max pooling 2×2 | 32×32×32 | 0 |
| 3 | Conv 3×3, 64 filters, ReLU | 32×32×64 | 18,496 |
| 4 | Max pooling 2×2 | 16×16×64 | 0 |
| 5 | Conv 3×3, 128 filters, ReLU | 16×16×128 | 73,856 |
| 6 | Max pooling 2×2 | 8×8×128 | 0 |
| 7 | Global average pooling | 128 | 0 |
| 8 | Dense 128, ReLU | 128 | 16,512 |
| 9 | Dropout 0.5 | 128 | 0 |
| 10 | Dense 1 (sigmoid in Keras, raw logit in PyTorch) | 1 | 129 |
| | **Total** | | **109,889** |

Global average pooling makes the parameter count independent of the input size.

#### Keras

```python
import keras
from keras import layers

def conv_block(filters):
    return [layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal"),
            layers.MaxPooling2D()]

def build_keras_cnn(input_shape=(64, 64, 3)):
    return keras.Sequential([
        layers.Input(shape=input_shape),
        augment,                                     # from Section 2.5, active only during training
        layers.Rescaling(1.0 / 255),
        *conv_block(32), *conv_block(64), *conv_block(128),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid", kernel_initializer="he_normal"),
    ])
```

#### PyTorch

```python
import torch.nn as nn

def conv_block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

class PyTorchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32), conv_block(32, 64), conv_block(64, 128),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1),
        )
        for m in self.modules():                     # same initialization as Keras: He normal, zero bias
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)    # logits, shape (batch,)
```

#### Equivalence check

```python
keras_params = model_keras.count_params()
torch_params = sum(p.numel() for p in model_torch.parameters() if p.requires_grad)
assert keras_params == torch_params == 109_889
```

#### How the two styles differ

| Aspect | Keras | PyTorch |
|---|---|---|
| **Style** | Declarative: list the layers in `Sequential` | Object-oriented: define a class with `forward()` |
| **Input channels / features** | Inferred from the previous layer | Given explicitly (`Conv2d(3, 32, ...)`, `Linear(128, 128)`) |
| **Activation** | An argument of the layer (`activation="relu"`) | A separate module (`nn.ReLU()`) |
| **Padding** | `padding="same"` | `padding=1` (explicit number) |
| **Flatten / pooling** | `GlobalAveragePooling2D()` | `AdaptiveAvgPool2d(1)` + `Flatten()` |
| **Initialization** | `kernel_initializer="he_normal"` per layer | A loop over modules |
| **Final layer** | Sigmoid inside the model | Raw logit; sigmoid lives in the loss |

### 2.7 Training Workflow: Keras vs. PyTorch

| Step | Keras | PyTorch |
|---|---|---|
| **Configure training** | `model.compile(optimizer, loss, metrics)` | Create `criterion` and `optimizer` objects |
| **Training loop** | `model.fit(...)` | Write the loop: forward, loss, `zero_grad`, `backward`, `step` |
| **Train / eval modes** | Automatic | `model.train()` and `model.eval()` |
| **Validation** | `validation_data=val_ds` | A second loop under `torch.no_grad()` |
| **Early stopping** | `EarlyStopping` callback | A patience counter in the loop |
| **Checkpointing** | `ModelCheckpoint` callback | `torch.save(model.state_dict(), ...)` |
| **Learning-rate schedule** | `ReduceLROnPlateau` callback | `scheduler.step(val_loss)` |
| **Device placement (GPU)** | Automatic | `.to(device)` for the model and every batch |
| **Predictions** | `model.predict(ds)` | A loop with `torch.no_grad()` and `torch.sigmoid` |
| **Metrics** | Built in (`Precision`, `Recall`, `AUC`) | Computed by us (`scikit-learn` or `torchmetrics`) |
| **Amount of code** | Less | More, but full control |
| **Debugging** | Harder to inspect inside `fit` | Easy to print and inspect inside the loop |

#### Keras

```python
model = build_keras_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-8),   # same epsilon as PyTorch
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.Precision(name="precision"),
             keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")],
)
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("results/keras/best.keras", monitor="val_loss", save_best_only=True),
]
history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)
```

#### PyTorch

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PyTorchCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8)

best_val, patience, wait = float("inf"), 5, 0
for epoch in range(30):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            val_loss += criterion(model(x), y).item() * len(y)
    val_loss /= len(val_loader.dataset)

    if val_loss < best_val:                       # checkpoint the best model
        best_val, wait = val_loss, 0
        torch.save(model.state_dict(), "results/pytorch/best.pt")
    else:                                         # early stopping
        wait += 1
        if wait >= patience:
            break

model.load_state_dict(torch.load("results/pytorch/best.pt"))      # restore the best weights
```

#### Saving predictions for the shared evaluation

```python
import numpy as np
import pandas as pd

# Keras (test_ds was created with shuffle=False)
y_prob = model.predict(test_ds).ravel()
y_true = np.concatenate([y.numpy().ravel() for _, y in test_ds]).astype(int)
pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv("results/keras/predictions.csv", index=False)

# PyTorch (test_loader was created with shuffle=False)
model.eval()
probs, labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu())
        labels.append(y)
y_prob = torch.cat(probs).numpy()
y_true = torch.cat(labels).numpy().astype(int)
pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).to_csv("results/pytorch/predictions.csv", index=False)
```

---

### 2.8 Evaluating the Classifier

All metrics are computed from the **saved test-set probabilities** with **one shared script**, so both frameworks are judged by exactly the same code. The positive class is **agricultural**.

#### The confusion matrix

| | **Predicted agricultural** | **Predicted non-agricultural** |
|---|---|---|
| **Actually agricultural** | True positive (TP) | False negative (FN) |
| **Actually non-agricultural** | False positive (FP) | True negative (TN) |

Every other metric below is computed from these four counts (or, for ROC and AP, from the probabilities directly).

#### The metrics

| Metric | Formula | The question it answers | Watch out for |
|---|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | How often is the model right overall? | Misleading when classes are imbalanced |
| **Precision** | $\frac{TP}{TP + FP}$ | Of the tiles predicted agricultural, how many really are? | Ignores missed farmland |
| **Recall** (sensitivity, TPR) | $\frac{TP}{TP + FN}$ | Of the real agricultural tiles, how many did we find? | Ignores false alarms |
| **F1 score** | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | One number that balances precision and recall | Hides which of the two is weak |
| **False positive rate** | $\frac{FP}{FP + TN}$ | Of the non-agricultural tiles, how many were wrongly flagged? | Used for the ROC curve |
| **ROC-AUC** | Area under the TPR-vs-FPR curve | How well does the model rank agricultural tiles above non-agricultural ones, at **any** threshold? | Can look optimistic under heavy class imbalance |
| **Average precision (AP)** | Area under the precision-recall curve | How good are precision and recall across thresholds, focused on the positive class? | Depends on class balance |
| **IoU (Jaccard index)** | $\frac{TP}{TP + FP + FN}$ | How much do the predicted and true agricultural sets overlap? | Ignores true negatives |
| **mAP** | Mean of AP over classes | AP averaged over the classes | See the note below |

> **About IoU and mAP.** These two metrics come from **object detection and segmentation**, where IoU measures the overlap between predicted and true boxes or masks, and mAP averages precision over classes (and, in detection, over IoU thresholds). For **image-level binary classification** they reduce to simpler quantities: IoU becomes the Jaccard index of the positive class (formula above), and mAP becomes the average of AP over the two classes. We report them for completeness, but **precision, recall, F1, and ROC-AUC are the primary metrics** here. They become truly central if the project is extended to segmenting farmland pixel by pixel.

#### ROC and AUC

The **ROC curve** plots the true positive rate (recall) against the false positive rate while the decision threshold moves from 1 to 0.

| ROC-AUC | Meaning |
|---|---|
| 0.5 | No better than random guessing |
| 0.5 to 1.0 | Better ranking; closer to 1 is better |
| 1.0 | Perfect ranking |

AUC can be read as: *the probability that a randomly chosen agricultural tile gets a higher score than a randomly chosen non-agricultural tile.* Unlike precision and recall, it does **not** depend on the threshold.

#### Precision and the decision threshold

Precision and recall depend on the threshold. Raising it makes the model more cautious:

| Threshold | Precision | Recall | Effect on the forecast |
|---|---|---|---|
| **Lower** | Falls | Rises | Fewer missed farms, more false alarms (risk of over-forecasting) |
| **Higher** | Rises | Falls | Fewer false alarms, more missed farms (risk of under-forecasting) |

Because we prioritize precision, we can choose the threshold **on the validation set** for a target precision, then apply it unchanged to the test set:

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_val_true, y_val_prob)
ok = np.where(precision[:-1] >= 0.95)[0]               # thresholds that reach 95% precision
chosen_threshold = thresholds[ok[0]] if len(ok) else 0.5   # smallest one keeps recall highest
```

#### The shared evaluation script

```python
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score, jaccard_score)

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "iou_jaccard": jaccard_score(y_true, y_pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }

results = {}
for name in ["keras", "pytorch"]:
    df = pd.read_csv(f"results/{name}/predictions.csv")
    results[name] = evaluate(df["y_true"].values, df["y_prob"].values)
print(pd.DataFrame(results))
```

```python
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots()
for name in ["keras", "pytorch"]:
    df = pd.read_csv(f"results/{name}/predictions.csv")
    RocCurveDisplay.from_predictions(df["y_true"], df["y_prob"], name=name, ax=ax)
fig.savefig("results/figures/roc_keras_vs_pytorch.png", bbox_inches="tight")
```

#### Which metric to look at when

| Question | Metric |
|---|---|
| Are we over-counting farmland? | **Precision** |
| Are we missing farmland? | **Recall** |
| Do we need one balanced number? | **F1** |
| Is the model good regardless of threshold? | **ROC-AUC** and **AP** |
| Where exactly are the mistakes? | **Confusion matrix** |
| Is the model good overall (balanced classes only)? | **Accuracy** |

### 2.9 Comparing Keras and PyTorch

Fill these tables with the final results (all values are **placeholders** until the experiments are run).

#### a) Test-set metrics (mean ± standard deviation over 3 seeds)

| Metric | Keras | PyTorch | Difference |
|---|---|---|---|
| Accuracy | TBD | TBD | TBD |
| **Precision** | TBD | TBD | TBD |
| Recall | TBD | TBD | TBD |
| F1 score | TBD | TBD | TBD |
| ROC-AUC | TBD | TBD | TBD |
| Average precision | TBD | TBD | TBD |
| IoU (Jaccard) | TBD | TBD | TBD |

#### b) Confusion matrices (seed 42, threshold 0.5)

| Keras | Predicted agricultural | Predicted non-agricultural |
|---|---:|---:|
| **Actually agricultural** | TBD | TBD |
| **Actually non-agricultural** | TBD | TBD |

| PyTorch | Predicted agricultural | Predicted non-agricultural |
|---|---:|---:|
| **Actually agricultural** | TBD | TBD |
| **Actually non-agricultural** | TBD | TBD |

#### c) Training behavior

| Measure | Keras | PyTorch |
|---|---|---|
| Trainable parameters | 109,889 | 109,889 |
| Epochs actually run (early stopping) | TBD | TBD |
| Epoch of the best validation loss | TBD | TBD |
| Training time per epoch | TBD | TBD |
| Best validation loss | TBD | TBD |

#### d) Workflow comparison (our observations)

| Aspect | Keras | PyTorch |
|---|---|---|
| Lines of code for the full pipeline | TBD | TBD |
| Time to a first working model | TBD | TBD |
| Debugging experience | TBD | TBD |
| Flexibility for custom behavior | TBD | TBD |
| Handling of data loading and augmentation | TBD | TBD |

#### How to interpret the comparison

| Observation | Conclusion |
|---|---|
| The difference between frameworks is **smaller than the seed-to-seed standard deviation** | No meaningful difference. Both implementations agree, which **validates the models** (Goal 1). |
| One framework is **clearly and consistently** better | Look for a hidden difference first (initialization, augmentation, preprocessing; see the table in Section 2.2) before concluding that the framework is better. |
| Results differ **a lot** between frameworks | Suspect a bug: check label order, preprocessing, augmentation on validation/test data, and `train`/`eval` modes. |

### 2.10 Conclusions

Template for the final write-up:

1. **Performance:** do both CNNs reach comparable precision, recall, F1, and ROC-AUC? *(TBD)*
2. **Validation:** does the agreement between the two independent implementations increase our trust in the results? *(TBD)*
3. **Workflow:** which framework was faster to develop with, and which gave more control? *(TBD)*
4. **Business value:** how many agricultural tiles are correctly identified, and what does the false-positive rate mean for the fertilizer forecast? *(TBD)*

**Limitations to state honestly:**

- One dataset, and image-level labels only (no pixel-level mapping)
- If tile locations are known, **random splits can overestimate performance**, because neighboring tiles look alike. Splitting by geographic region is a stricter test.
- Hyperparameters were tuned on a small search space
- The sales forecast itself is not evaluated here

**Possible next steps:**

- Transfer learning with a pretrained backbone (for example, ResNet)
- Multi-spectral bands (such as near-infrared) for better vegetation signals
- Segmentation of farmland, where IoU and mAP become the main metrics
- Connecting the classifier's output to the fertilizer sales forecasting model

---

## Appendix A: Data and Setup

| Item | Details |
|---|---|
| **Dataset** | `[name, source link, license]` |
| **Size** | `[number of images, image size, number per class]` |
| **Class balance** | `[agricultural : non-agricultural]` |
| **Download** | `[instructions]` (raw data is **not** committed to the repository) |

**Requirements:** the code uses the **Keras 3** API (TensorFlow 2.16 or newer as the backend, or another supported backend) and a recent PyTorch with torchvision (the `transforms.v2` API).

```text
keras
tensorflow
torch
torchvision
scikit-learn
pandas
numpy
pillow
matplotlib
```
**Run order:** the notebooks in `notebooks/`, from `01` to `05`.

## Appendix B: Reproducibility Checklist

- [ ] Fixed random seed (42, 43, 44) in Python, NumPy, Keras (`keras.utils.set_random_seed`), and PyTorch (`torch.manual_seed`)
- [ ] Splits created once, stored as CSV files, and reused by both frameworks
- [ ] Positive class (`agricultural` = 1) verified in both pipelines
- [ ] Augmentation applied to **training data only**
- [ ] Explicit initializers, Adam epsilon, padding, and rotation fill in both frameworks
- [ ] Parameter counts match (109,889)
- [ ] Validation and test loaders use `shuffle=False`
- [ ] All metrics computed by the same script from saved predictions
- [ ] Test set used **once**, after the hyperparameters were chosen
- [ ] Library versions recorded in `requirements.txt`

## Appendix C: Suggested Presentation Outline

| # | Slide | Key message | Visual |
|---:|---|---|---|
| 1 | Title and problem | Classify satellite tiles as agricultural or not | Example images of both classes |
| 2 | Business value | Farmland estimates feed fertilizer sales forecasts; precision guards against over-forecasting | Error-effect table |
| 3 | Data overview | Source, size, class balance, splits | Class-balance chart |
| 4 | Loading methods | Bulk (memory-based) vs. sequential (generator-based) | Comparison table and memory formula |
| 5 | Efficient pipelines | Custom generators vs. built-in utilities, in both frameworks | Side-by-side code |
| 6 | Augmentation in PyTorch | Label-preserving transforms for satellite images | Grid of augmented samples |
| 7 | CNN architecture | Same architecture in both frameworks (109,889 parameters) | Layer table or diagram |
| 8 | Fair-comparison design | Same data, batch size, optimizer, learning rate, epochs | Shared-configuration table |
| 9 | Hyperparameters | Optimization, regularization, batch size, epochs | Tuning-log summary |
| 10 | Evaluation metrics | Precision, recall, F1, ROC-AUC (plus IoU and mAP in context) | Confusion matrix |
| 11 | Results | Keras vs. PyTorch metrics, mean ± std | Metrics table and ROC overlay |
| 12 | Workflow comparison | Where Keras and PyTorch differ in practice | Workflow table |
| 13 | Conclusions and next steps | What we learned, limitations, extensions | Bullet summary |
