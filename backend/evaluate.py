import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from backend.ml.model import SkinCancerResNet

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = SkinCancerResNet(num_classes=2).to(device)
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.eval()

# define class names (adjust if you used more labels)
class_names = ["Benign", "Malignant"]

# image transforms (must match your training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet values
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return class_names[predicted.item()]

if __name__ == "__main__":
    test_image_path = "dataset/melanoma_cancer_dataset/test/malignant/melanoma_10109.jpg"  # replace with your image file
    prediction = predict_image(test_image_path)
    print(f"Prediction: {prediction}")