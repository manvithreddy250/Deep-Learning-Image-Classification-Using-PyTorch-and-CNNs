from pathlib import Path
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import SmallCNN

BASE = Path(__file__).parent
st.set_page_config(page_title="Image Classifier", page_icon="🖼️")
st.title("🖼️ Deep Learning Image Classifier")
st.caption("CIFAR-10 demo model. It works best with small, centered everyday-object images similar to the dataset.")
weights = BASE / "cifar10_cnn.pth"
if not weights.exists():
    st.warning("No trained model found. Run `python train.py` first; the first run downloads CIFAR-10.")
    st.stop()
checkpoint = torch.load(weights, map_location="cpu")
classes = checkpoint["classes"]
model = SmallCNN(num_classes=len(classes))
model.load_state_dict(checkpoint["state_dict"])
model.eval()
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])
upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if upload:
    image = Image.open(upload).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)
    values, indices = torch.topk(probs, k=3)
    st.subheader("Top predictions")
    for value, idx in zip(values.tolist(), indices.tolist()):
        st.write(f"**{classes[idx]}** — {value:.1%}")
    st.caption("Predictions are model estimates and may be wrong, especially for images unlike CIFAR-10.")
