# Forest Path Segmentation — HOG Texture

Segments a forest image into textural regions (path, grass, trees)
using HOG descriptors + KMeans / SVM classification.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Pipeline
1. Load + preprocess image
2. Extract image patches
3. Compute HOG descriptor per patch
4. Cluster patches with KMeans
5. Visualize segmentation map