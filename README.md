# Deep Learning Image Classification

A PyTorch CNN trained on CIFAR-10, with a Streamlit interface for top-three predictions on uploaded images.

## Run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```
The first training run downloads CIFAR-10 and may take a while. A GPU is optional; CPU training will be slower. The model checkpoint is generated locally and is excluded from Git by `.gitignore` patterns.

## Classes
CIFAR-10 includes airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

## Limitations
This is a small educational baseline, not a production-grade classifier. Training time and accuracy vary by hardware and settings.
