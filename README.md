# Food Recognition

## Overview
This repository contains an exploratory project for food classification using deep learning. The goal is to identify the type of food in an image using modern convolutional and transformer-based networks. The work focuses on building reproducible experiments (notebooks and scripts) to preprocess the Food-101 dataset, train models, evaluate results, and compare trade-offs between accuracy and compute.

Applications:
- Menu digitization
- Dietary monitoring
- Self-checkout systems
- Health tracking
- Food quality control

## Dataset
We use the Food-101 dataset (ETH Zurich):

- Total images: 101,000  
- Classes: 101 food categories  
- Split: 750 training + 250 testing images per class  
- Original image size: 512×512 (commonly resized to 224×224 for training)

Download: http://www.vision.ee.ethz.ch/datasets_extra/food-101/  
Citation:
Bossard, Lukas; Guillaumin, Matthieu; Van Gool, Luc. "Food-101 – Mining Discriminative Components with Random Forests." ECCV 2014.

BibTeX:
```bibtex
@inproceedings{bossard2014food,
  title={Food-101--mining discriminative components with random forests},
  author={Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle={European conference on computer vision},
  pages={446--461},
  year={2014},
  organization={Springer}
}
```

Dataset layout (after extraction)
```
food-101/
  images/
    apple_pie/
      0001.jpg
      ...
    ...
  meta/
    train.txt
    test.txt
```

## Models Evaluated
The notebook explores and compares the following architectures:
- ResNet50 (transfer learning & fine-tuning)
- EfficientNet (B0/B3 depending on resources)
- ConvNeXt
- ViT (used in ensembles)
- Ensembles: ResNet50 + ViT, ConvNeXt + EfficientNet

**ResNet50 + ViT: A Hybrid Approach to Computer Vision**

This project implements a **hybrid deep learning model** that combines the strengths of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) for image analysis. The architecture leverages ResNet50 for local spatial feature extraction and ViT’s transformer-based self-attention for global contextual understanding. This synergy enables robust and comprehensive modeling of both fine details and long-range dependencies in computer vision tasks.

**Key Model Highlights:**

- **CNN Component (ResNet50):**  
  Extracts hierarchical local features from input images—edges, textures, and spatial relationships—using convolutional layers.

- **ViT Component:**  
  Applies transformer self-attention to the CNN feature map, enabling global context modeling and learning of long-range dependencies across the entire image.

- **Synergy:**  
  By merging both approaches, the model benefits from the spatial precision of CNNs and the contextual depth of transformers, resulting in superior performance and robust image understanding.

***


Summary:
- ResNet50, EfficientNet, and ConvNeXt: strong performance, faster training.
- Ensembles (ResNet50 + ViT, ConvNeXt + EfficientNet): best results but significantly longer training and higher resource use.
Recommendation: use ResNet50 or EfficientNet for a balance of accuracy and efficiency.

## Requirements
Python 3.8+ recommended.

Required Python packages:
- pandas
- numpy
- scikit-learn
- Pillow
- torch
- torchvision
- matplotlib
- torchinfo
- tqdm
- pathlib
- seaborn

Install with pip:
```bash
pip install pandas numpy scikit-learn pillow torch torchvision matplotlib torchinfo tqdm pathlib seaborn
```

(Optional) Create a conda environment:
```bash
conda create -n foodrec python=3.9
conda activate foodrec
pip install -r requirements.txt
```
(If you add a requirements.txt file, list the above packages.)

## Notebook / Scripts
Primary artifact: Jupyter notebook (e.g., `food_recognition.ipynb`) that contains steps for:
1. Loading and inspecting the Food-101 dataset
2. Preprocessing and augmentation (resize to 224×224, normalization)
3. Building PyTorch datasets and dataloaders
4. Training and validation loops
5. Model definitions and transfer learning setup
6. Evaluation (accuracy, top-5 accuracy, confusion matrix)
7. Visualization of predictions and misclassifications

If there are separate training scripts (e.g., `train.py`) and evaluation scripts (`eval.py`), use them as:
```bash
python train.py --data /path/to/food-101 --model resnet50 --batch-size 32 --epochs 20 --img-size 224
python eval.py  --data /path/to/food-101 --model resnet50 --weights weights/resnet50_final.pth
```

Adjust flags according to the script arguments included in the repo.

## How to Run (quick)
1. Download and extract Food-101.
2. Open the notebook in Jupyter:
```bash
jupyter lab    # or jupyter notebook
```
3. Update dataset path variables in the notebook (e.g., DATA_DIR = "/path/to/food-101").
4. Run cells sequentially to preprocess, train, and evaluate.

Tips:
- Use a GPU for training (CUDA + compatible PyTorch build).
- For faster iterations, freeze backbone and train the classifier head first, then unfreeze and fine-tune with a smaller learning rate.
- Use mixed precision (torch.cuda.amp) to reduce memory and speed up training if available.

## Training Recommendations & Hyperparameters
- Image size: 224×224 (224 recommended; try 320 or 384 for larger models if resources allow)
- Batch size: 32–128 (depending on GPU memory)
- Optimizer: AdamW or SGD+Momentum
- Learning rate: 1e-3 (head), 1e-4 to 1e-5 (fine-tuning backbone)
- Weight decay: 1e-4
- Scheduler: CosineAnnealingLR or StepLR
- Epochs: 20–50 (more for fine-tuning / larger models)
- Augmentations: random crop, random horizontal flip, color jitter, normalization to ImageNet mean/std

## Metrics and Evaluation
- Top-1 accuracy (primary)
- Top-5 accuracy (helpful for fine-grained classes)
- Confusion matrix to inspect per-class errors
- Per-class precision/recall/F1 for imbalanced analysis (though classes here are balanced)

## Results (example summary)
- ResNet50: strong baseline, good speed/accuracy trade-off
- EfficientNet: high accuracy with parameter efficiency (choose variant by compute: B0–B3 are good starts)
- ConvNeXt: competitive results for modern conv architectures
- Ensembles: best single-run accuracy, increased inference & training cost

(Include your actual numeric results and plots from the notebook here once you run experiments.)

## Reproducibility
- Set random seeds for Python, NumPy, and torch for deterministic runs where possible.
- Log experiment hyperparameters and model checkpoints (use TensorBoard or Weights & Biases for full experiment tracking).
- Save the exact code and commit hash used for each reported result.

