from glob import glob
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import models
import os

# === Model definition ===
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet50(weights=None)  # Set pretrained=False for a custom-trained model
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)  # assuming 4 classes

    def forward(self, x):
        return self.model(x)

net = Net()

# === Transform function ===
def get_valid_transforms():
    from albumentations import Compose, Normalize, Resize
    from albumentations.pytorch import ToTensorV2
    return Compose([
        Resize(224, 224),  # Resize to 224x224 for ResNet50 input size
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet normalization
        ToTensorV2(),
    ])

# === Dataset class for single image ===
class SingleImageDataset(Dataset):
    def __init__(self, image_path, transforms=None):
        super().__init__()
        self.image_path = image_path
        self.transforms = transforms

    def __getitem__(self, index):
        image_name = os.path.basename(self.image_path)  # Handle different path separators
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        return image_name, image

    def __len__(self):
        return 1  # Single image

# === Load model checkpoint ===
checkpoint = torch.load("C:\\Users\\Aniruddh Rajagopal\\Downloads\\best-checkpoint-023epoch.bin\\best-checkpoint-023epoch.bin", map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['model_state_dict'], strict=False)
net.eval()

# === Prediction function ===
def predict_single_image(image_path):
    dataset = SingleImageDataset(image_path=image_path, transforms=get_valid_transforms())
    image_name, image = dataset[0]  # Directly get the image and its name

    # Add batch dimension
    image = image.unsqueeze(0)  # Convert to shape (1, C, H, W)

    with torch.no_grad():
        y_pred = net(image)
        y_pred = nn.functional.softmax(y_pred, dim=1).cpu().numpy()

        # Get the class with the highest probability
        predicted_class = np.argmax(y_pred, axis=1)
        label = predicted_class[0]

    return image_name, label

# === Test ===
image_path = "C:\\Users\\Aniruddh Rajagopal\\Downloads\\00001_normal.jpg"  # Replace with your actual image path
image_name, label = predict_single_image(image_path)
print(f"Image: {image_name} — Predicted Label: {label}")
