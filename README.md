# HW1 Image Classifier

Name: Imrose Batterywala

Student ID: 314540010

This homework focused on fine-tuning timm backbones for a 100-class image classification task.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Train

```bash
python3 train.py --data-dir data --output-dir artifacts --epochs 100 --batch-size 8
```

## Predict

```bash
python3 predict.py --checkpoint artifacts/best_model.pt --test-dir data/test --output artifacts/predictions.csv
```

## Leaderboard

### Snapshot (1)

![1](./snapshots/l_1.png)

### Snapshot (2)

![2](./snapshots/l_2.png)

## Artifacts

### Confusion Matrix

![Confusion Matrix](./snapshots/confusion_matrix_val.png)

### Training Curve

![Training Curve](./snapshots/training_curve.png)
