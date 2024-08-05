import os
import json
import datetime
import argparse
import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
import numpy as np

# Function to get transformation for input images
def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform

# Function to load the model
def load_model(model_path, num_classes):
    model_choice = 'resnet18'  # Change this if you used a different model (resnet18, resnet50, vgg16)
    use_dropout = False  # Change this based on your model configuration

    if model_choice == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
    elif model_choice == 'vgg16':
        model = models.vgg16(weights='IMAGENET1K_V1')
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:  # Default to resnet18
        model = models.resnet18(weights='IMAGENET1K_V1')

    if model_choice in ['resnet50', 'resnet18']:
        if use_dropout:
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.fc.in_features, num_classes)
            )
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

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
                    'class_probabilities': {classes[j]: float(probs[i][j]) for j in range(len(classes))},
                    'predicted_class': classes[preds[i]]
                }
                results.append(result)
    return results

# Main function
def main():
    parser = argparse.ArgumentParser(description='Predict images using a saved model.')
    parser.add_argument('--model_path', type=str, default='path/to/saved_model.pth', help='Path to the saved model.')
    parser.add_argument('--input_folder', type=str, default='path/to/input_images', help='Folder containing input images.')
    parser.add_argument('--results_folder', type=str, default='results', help='Folder to save results.')
    
    args = parser.parse_args()
    
    model_path = args.model_path
    input_folder = args.input_folder
    results_folder = args.results_folder

    transform = get_transform()
    input_dataset = ImageFolder(root=input_folder, transform=transform)
    input_dataset.samples = [(path, 0) for path, _ in input_dataset.samples]  # Remove labels
    dataloader = DataLoader(input_dataset, batch_size=1, shuffle=False)

    num_classes = len(input_dataset.classes)
    model = load_model(model_path, num_classes)

    classes = input_dataset.classes
    results = make_predictions(model, dataloader, classes)

    # Create a timestamped folder for results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    result_folder = os.path.join(results_folder, timestamp)
    os.makedirs(result_folder, exist_ok=True)

    # Save results to a file
    result_file_path = os.path.join(result_folder, 'predictions.json')
    with open(result_file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f'Results saved to {result_file_path}')

if __name__ == "__main__":
    main()
