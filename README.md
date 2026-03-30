# HW1 Image Classifier

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

## Artifacts

![Confusion Matrix](./artifacts/confusion_matrix_val.png)

![Training Curve](./artifacts/training_curve.png)
