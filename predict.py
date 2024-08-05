import os
import json
import datetime
import argparse
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil

# Custom dataset for loading images from a folder
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Function to get transformation for input images
def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform

# Function to load the model
def load_model(model_path):
    model_choice = 'resnet18'  # Change this if you used a different model (resnet18, resnet50, vgg16)

    if model_choice == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
    elif model_choice == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1')
    else:  # Default to resnet18
        model = models.resnet18(weights='IMAGENET1K_V1')

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # Remove 'model.' prefix if exists
    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

    # Get the number of output classes
    num_classes = state_dict['fc.weight'].shape[0]
    
    # Adjust the final layer to match the number of classes
    if model_choice in ['resnet50', 'resnet18']:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_choice == 'vgg16':
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    model.load_state_dict(state_dict)
    model.eval()
    return model, num_classes

# Function to make predictions on input images
def make_predictions(model, dataloader, classes):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    results = []

    with torch.no_grad():
        for inputs, paths in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            for i in range(len(paths)):
                result = {
                    'filename': os.path.basename(paths[i]),
                    'predicted_class': classes[preds[i]]
                }
                for j, class_name in enumerate(classes):
                    result[f'probability_{class_name}'] = float(probs[i][j])
                results.append(result)
    return results

# Main function
def main():
    parser = argparse.ArgumentParser(description='Predict images using a saved model.')
    parser.add_argument('--model_path', type=str, default='E:/GitHub/FireAndSmokeDetector/results/20240805_192230/run_1/model.pth', help='Path to the saved model.')
    parser.add_argument('--input_folder', type=str, default='PhotosToPredict/input', help='Folder containing input images.')
    parser.add_argument('--results_folder', type=str, default='PhotosToPredict/output', help='Folder to save results.')
    
    args = parser.parse_args()
    
    model_path = args.model_path
    input_folder = args.input_folder
    results_folder = args.results_folder

    transform = get_transform()
    input_dataset = ImageDataset(image_folder=input_folder, transform=transform)
    dataloader = DataLoader(input_dataset, batch_size=1, shuffle=False)

    model, num_classes = load_model(model_path)

    # Define your actual class names here
    classes = ['bothFireAndSmoke', 'fire', 'neitherFireNorSmoke', 'smoke']  # Update with your actual class names

    results = make_predictions(model, dataloader, classes)

    # Create a timestamped folder for results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_folder = os.path.join(results_folder, timestamp)
    os.makedirs(result_folder, exist_ok=True)

    # Save results to an Excel file
    result_file_path = os.path.join(result_folder, 'predictions.xlsx')
    results_df = pd.DataFrame(results)
    results_df.to_excel(result_file_path, index=False)

    # Copy images to the output folder with the class label prefix
    for result in results:
        src_path = os.path.join(input_folder, result['filename'])
        dst_filename = f"{result['predicted_class']}_{result['filename']}"
        dst_path = os.path.join(result_folder, dst_filename)
        shutil.copyfile(src_path, dst_path)

    print(f'Results saved to {result_file_path}')
    print(f'Images saved with class label prefixes in {result_folder}')

if __name__ == "__main__":
    main()
