import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 🔹 Class names (MUST match training order)
class_names = [
    "aloo_tikki","bhatura","chana_masala","daal_puri",
    "jalebi","kachori","kuzhi_paniyaram","lassi",
    "litti_chokha","poha","rabri","unni_appam"
]

# ['aloo_tikki', 'bhatura', 'chana_masala', 'daal_puri', 'jalebi', 'kachori', 'kuzhi_paniyaram', 'lassi', 'litti_chokha', 'poha', 'rabri', 'unni_appam']

# 🔹 Load model
def load_model(model_path):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 12)

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# 🔹 Image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# 🔹 Prediction function
def predict(image, model):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return class_names[pred.item()], conf.item()