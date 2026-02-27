# Cataract Screening Pipeline

A full machine learning pipeline for automated cataract and ocular disease screening
from retinal fundus images.

## Dataset

Download from Kaggle: https://www.kaggle.com/datasets/jr2ngb/cataractdataset

Place the dataset so it matches this structure:
```
data/
└── dataset/
    ├── 1_normal/
    ├── 2_cataract/
    ├── 2_glaucoma/
    └── 3_retina_disease/
```

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline Scripts — Run in Order

| Script | Phase | Description |
|--------|-------|-------------|
| `01_eda.py` | 1 | Exploratory data analysis & preprocessing audit |
| `02_preprocess.py` | 1 | Image preprocessing & train/val/test split |
| `03_classify.py` | 2 | Multi-class CNN classifier (EfficientNet-B3) |
| `04_evaluate.py` | 2 | Full evaluation: metrics, Grad-CAM, ROC curves |
| `05_segment.py` | 3 | U-Net optic disc/lens segmentation + CDR biomarker |
| `06_augment.py` | 4 | Conditional GAN for synthetic data augmentation |
| `07_app.py` | 5 | Streamlit inference demo (all modules integrated) |

### Quick Run (all phases)
```bash
python 01_eda.py
python 02_preprocess.py
python 03_classify.py
python 04_evaluate.py
python 05_segment.py
python 06_augment.py
streamlit run 07_app.py
```

## Outputs

All outputs are saved to `outputs/`:
- `eda/`         — EDA plots and class distribution reports
- `processed/`  — Preprocessed dataset splits
- `classifier/` — Saved model weights and training curves
- `evaluation/` — Metrics, Grad-CAM visualisations, ROC curves
- `segmentation/`— Segmentation masks and CDR scores
- `augmented/`  — GAN-generated synthetic images
