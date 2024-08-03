import os
import json
import shutil
import random
from PIL import Image
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
import torch
from torch.nn.functional import cosine_similarity
import numpy as np

def calculate_similarity(feature1, feature2):
    feature1 = torch.tensor(feature1).flatten()
    feature2 = torch.tensor(feature2).flatten()
    similarity = cosine_similarity(feature1.unsqueeze(0), feature2.unsqueeze(0))
    return similarity.item()

def load_and_extract_features(folder, model, transform, device):
    print(f"Loading and extracting features from folder: {folder}")
    features = []
    file_list = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            image = Image.open(file_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model(image)['avgpool'].cpu().numpy().squeeze()
            features.append(feature)
            file_list.append((file, subfolder))
    print(f"Extracted features from {len(file_list)} images.")
    return features, file_list

def prepare_folders(train_folder, test_folder, dataset_folder):
    # Overwrite existing train and test folders
    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    
    print("Creating train and test folders...")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for subfolder in os.listdir(dataset_folder):
        os.makedirs(os.path.join(train_folder, subfolder), exist_ok=True)
        os.makedirs(os.path.join(test_folder, subfolder), exist_ok=True)

def similarity_based_sampling(features, file_list, similarity_threshold, max_similar_photos, model, device):
    train_files = []
    test_files = []

    # Progress tracking initialization
    total_files = len(file_list)

    # Iterate over the entire dataset
    for i, (file, subfolder) in enumerate(file_list):
        current_feature = features[i]

        def add_to_folder(target_folder):
            similar_count = 0
            for train_file, train_subfolder in target_folder:
                train_feature = features[file_list.index((train_file, train_subfolder))]
                similarity = calculate_similarity(current_feature, train_feature)
                
                if similarity > similarity_threshold:
                    similar_count += 1

            if similar_count < max_similar_photos:
                target_folder.append((file, subfolder))
                return True
            return False

        # Attempt to add to train_files first, if not possible, add to test_files
        if not add_to_folder(train_files):
            add_to_folder(test_files)
        
        # Print progress every 10%
        if (i + 1) % (total_files // 10) == 0:
            print(f"Similarity-based sampling progress: {((i + 1) / total_files) * 100:.1f}%")

    return train_files, test_files

def copy_files_to_folders(files, source_folder, dest_folder):
    for file, subfolder in files:
        shutil.copy(os.path.join(source_folder, subfolder, file), os.path.join(dest_folder, subfolder, file))

def random_sampling(dataset_folder, sampling_ratio, test_ratio):
    train_files = []
    test_files = []
    
    for subfolder in os.listdir(dataset_folder):
        files = os.listdir(os.path.join(dataset_folder, subfolder))
        sample_size = int(len(files) * sampling_ratio)
        sampled_files = random.sample(files, sample_size)
        random.shuffle(sampled_files)
        split_index = int(len(sampled_files) * (1 - test_ratio))
        train_files.extend([(file, subfolder) for file in sampled_files[:split_index]])
        test_files.extend([(file, subfolder) for file in sampled_files[split_index:]])

        # Print progress every 10%
        for i, _ in enumerate(sampled_files):
            if (i + 1) % (sample_size // 10) == 0:
                print(f"Random sampling progress for class '{subfolder}': {((i + 1) / sample_size) * 100:.1f}%")
    
    return train_files, test_files

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dataset_folder = config.get('dataset_folder', 'dataset')
    train_folder = config.get('train_folder', 'train')
    test_folder = config.get('test_folder', 'test')
    sampling_method = config.get('sampling_method', 'random')
    random_seed = config.get('random_seed', 42)
    similarity_threshold = config.get('similarity_threshold', 0.95)
    max_similar_photos = config.get('max_similar_photos', 2)
    test_ratio = config.get('test_ratio', 0.2)
    sampling_ratio = config.get('sampling_ratio', 0.4)

    return (dataset_folder, train_folder, test_folder, sampling_method, random_seed, 
            similarity_threshold, max_similar_photos, test_ratio, sampling_ratio)

def create_train_test_folders(config_path):
    (dataset_folder, train_folder, test_folder, sampling_method, random_seed, 
     similarity_threshold, max_similar_photos, test_ratio, sampling_ratio) = read_config(config_path)
    
    random.seed(random_seed)

    prepare_folders(train_folder, test_folder, dataset_folder)

    if sampling_method == 'similarity':
        print("Using similarity-based sampling...")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = models.resnet18(weights='IMAGENET1K_V1')
        model = create_feature_extractor(model, return_nodes={'avgpool': 'avgpool'})
        model = model.to(device)
        model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        features, file_list = load_and_extract_features(dataset_folder, model, transform, device)
        train_files, test_files = similarity_based_sampling(features, file_list, similarity_threshold, max_similar_photos, model, device)
    else:
        print("Using random sampling...")
        train_files, test_files = random_sampling(dataset_folder, sampling_ratio, test_ratio)
    
    print(f"Selected {len(train_files)} files for training and {len(test_files)} files for testing.")
    copy_files_to_folders(train_files, dataset_folder, train_folder)
    copy_files_to_folders(test_files, dataset_folder, test_folder)

    print("Data has been successfully split into train and test sets.")

def main():
    # Configuration file path
    config_path = 'config.json'
    
    # Create train and test folders
    create_train_test_folders(config_path)

if __name__ == "__main__":
    main()
