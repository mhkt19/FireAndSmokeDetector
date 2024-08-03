import os
import json
import random
import datetime
import numpy as np
import time
import tracemalloc
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import copy

# Define the model
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Classifier, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def train_and_evaluate(model, criterion, optimizer, scheduler, dataloaders, config):
    print("Starting training and evaluation...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0
    num_epochs = config.get('num_epochs', 15)
    patience = config.get('patience', 3)

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_counter = 0
            elif phase == 'test':
                patience_counter += 1

            # Calculate precision and recall
            precision = precision_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')

            # Save metrics for each epoch
            if phase == 'test':
                test_metrics = {
                    'epoch': epoch,
                    'loss': epoch_loss,
                    'accuracy': round(epoch_acc.item() * 100, 2),
                    'precision': round(precision * 100, 2),
                    'recall': round(recall * 100, 2)
                }
            else:
                train_metrics = {
                    'epoch': epoch,
                    'loss': epoch_loss,
                    'accuracy': round(epoch_acc.item() * 100, 2),
                    'precision': round(precision * 100, 2),
                    'recall': round(recall * 100, 2)
                }

        scheduler.step(epoch_loss)
        print()

        if patience_counter >= patience:
            print("Early stopping")
            break

    duration = time.time() - start_time

    print(f'Best test Acc: {round(best_acc.item() * 100, 2)}%')
    model.load_state_dict(best_model_wts)
    return model, duration, train_metrics, test_metrics

def map_to_binary_classes(labels, test_dataset):
    binary_labels = []
    for label in labels:
        if test_dataset.classes[label] == 'neitherFireNorSmoke':
            binary_labels.append(0)  # Neither fire nor smoke
        else:
            binary_labels.append(1)  # Both fire and smoke, fire, smoke
    return binary_labels

def test_model(model, dataloader, results_folder, dataset, run_index, phase):
    print(f"Starting model testing for run {run_index + 1}, phase {phase}...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    misclassified_images = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            for i in range(len(labels)):
                if labels[i] != predicted[i]:
                    misclassified_images.append((inputs[i], labels[i], predicted[i]))
    
    accuracy = round(100 * correct / total, 2)
    print(f'{phase} Accuracy: {accuracy}%')
    
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=dataset.classes)
    
    precision = round(precision_score(all_labels, all_preds, average='weighted') * 100, 2)
    recall = round(recall_score(all_labels, all_preds, average='weighted') * 100, 2)

    phase_folder = os.path.join(results_folder, f'run_{run_index + 1}', phase)
    os.makedirs(phase_folder, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{phase.capitalize()} Confusion Matrix')
    plt.savefig(os.path.join(phase_folder, 'confusion_matrix.png'))
    plt.close()
    
    with open(os.path.join(phase_folder, 'classification_report.txt'), 'w') as f:
        f.write(cr)
    
    misclassified_folder = os.path.join(phase_folder, 'misclassified')
    os.makedirs(misclassified_folder, exist_ok=True)
    for idx, (img_tensor, true_label, pred_label) in enumerate(misclassified_images):
        img = transforms.ToPILImage()(img_tensor.cpu())
        img.save(os.path.join(misclassified_folder, f'{idx}_true_{dataset.classes[true_label]}_pred_{dataset.classes[pred_label]}.png'))
    
    # Map to binary classes
    binary_labels = map_to_binary_classes(all_labels, dataset)
    binary_preds = map_to_binary_classes(all_preds, dataset)

    binary_accuracy = round(np.mean(np.array(binary_labels) == np.array(binary_preds)) * 100, 2)
    binary_cm = confusion_matrix(binary_labels, binary_preds)
    binary_cr = classification_report(binary_labels, binary_preds, target_names=['neitherFireNorSmoke', 'fire_and_smoke'])

    binary_precision = round(precision_score(binary_labels, binary_preds, average='weighted') * 100, 2)
    binary_recall = round(recall_score(binary_labels, binary_preds, average='weighted') * 100, 2)

    # Save binary classification report and confusion matrix
    with open(os.path.join(phase_folder, 'binary_classification_report.txt'), 'w') as f:
        f.write(binary_cr)

    plt.figure(figsize=(10, 8))
    sns.heatmap(binary_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['neitherFireNorSmoke', 'fire_and_smoke'], yticklabels=['neitherFireNorSmoke', 'fire_and_smoke'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{phase.capitalize()} Binary Confusion Matrix')
    plt.savefig(os.path.join(phase_folder, 'binary_confusion_matrix.png'))
    plt.close()

    # Save probabilities to file
    prob_file_path = os.path.join(phase_folder, 'probabilities.csv')
    with open(prob_file_path, 'w') as f:
        header = "Filename,Class_Prob_1,Class_Prob_2,Class_Prob_3,Class_Prob_4,Predicted_Class\n"
        f.write(header)
        for idx, (label, pred, prob) in enumerate(zip(all_labels, all_preds, all_probs)):
            filename = dataset.imgs[idx][0]
            prob_str = ",".join([f"{round(p, 2):.2f}" for p in prob])
            f.write(f"{os.path.basename(filename)},{prob_str},{dataset.classes[pred]}\n")

    return accuracy, cm, precision, recall, binary_accuracy, binary_cm, binary_precision, binary_recall

def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Extract configuration variables
    batch_size = config.get('batch_size', 32)
    learning_rate = config.get('learning_rate', 0.001)
    train_folder = config.get('train_folder', 'train')
    test_folder = config.get('test_folder', 'test')
    results_folder = config.get('results_folder', 'results')
    num_runs = config.get('num_runs', 10)

    # Set random seed for reproducibility
    random_seed = config.get('random_seed', 42)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load datasets
    print("Loading datasets...")
    train_dataset = ImageFolder(root=train_folder, transform=transform)
    test_dataset = ImageFolder(root=test_folder, transform=transform)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    }

    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Initialize metrics storage
    all_train_metrics = {
        'accuracies': [],
        'confusion_matrices': [],
        'precisions': [],
        'recalls': [],
        'binary_accuracies': [],
        'binary_confusion_matrices': [],
        'binary_precisions': [],
        'binary_recalls': [],
    }
    all_test_metrics = {
        'accuracies': [],
        'confusion_matrices': [],
        'precisions': [],
        'recalls': [],
        'binary_accuracies': [],
        'binary_confusion_matrices': [],
        'binary_precisions': [],
        'binary_recalls': [],
    }
    all_durations = []
    all_memory_usages = []

    # Create a timestamped folder for this series of runs
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    series_results_folder = os.path.join(results_folder, timestamp)
    os.makedirs(series_results_folder, exist_ok=True)

    for run_index in range(num_runs):
        print(f"Run {run_index + 1}/{num_runs}")
        # Define the model
        model = ResNet18Classifier(num_classes=len(train_dataset.classes))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

        tracemalloc.start()

        # Train and evaluate the model
        trained_model, duration, train_metrics, test_metrics = train_and_evaluate(model, criterion, optimizer, scheduler, dataloaders, config)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_usage = peak / (1024 * 1024)  # Convert to MB

        # Test the model and save results for training phase
        train_phase_results = test_model(trained_model, dataloaders['train'], series_results_folder, train_dataset, run_index, 'train')
        # Test the model and save results for testing phase
        test_phase_results = test_model(trained_model, dataloaders['test'], series_results_folder, test_dataset, run_index, 'test')

        # Store metrics
        all_train_metrics['accuracies'].append(train_phase_results[0])
        all_train_metrics['confusion_matrices'].append(train_phase_results[1])
        all_train_metrics['precisions'].append(train_phase_results[2])
        all_train_metrics['recalls'].append(train_phase_results[3])
        all_train_metrics['binary_accuracies'].append(train_phase_results[4])
        all_train_metrics['binary_confusion_matrices'].append(train_phase_results[5])
        all_train_metrics['binary_precisions'].append(train_phase_results[6])
        all_train_metrics['binary_recalls'].append(train_phase_results[7])

        all_test_metrics['accuracies'].append(test_phase_results[0])
        all_test_metrics['confusion_matrices'].append(test_phase_results[1])
        all_test_metrics['precisions'].append(test_phase_results[2])
        all_test_metrics['recalls'].append(test_phase_results[3])
        all_test_metrics['binary_accuracies'].append(test_phase_results[4])
        all_test_metrics['binary_confusion_matrices'].append(test_phase_results[5])
        all_test_metrics['binary_precisions'].append(test_phase_results[6])
        all_test_metrics['binary_recalls'].append(test_phase_results[7])

        all_durations.append(duration)
        all_memory_usages.append(memory_usage)

    # Calculate average metrics
    def calculate_average_metrics(metrics):
        return {
            'accuracy': np.mean(metrics['accuracies']),
            'confusion_matrix': np.mean(metrics['confusion_matrices'], axis=0),
            'precision': np.mean(metrics['precisions']),
            'recall': np.mean(metrics['recalls']),
            'binary_accuracy': np.mean(metrics['binary_accuracies']),
            'binary_confusion_matrix': np.mean(metrics['binary_confusion_matrices'], axis=0),
            'binary_precision': np.mean(metrics['binary_precisions']),
            'binary_recall': np.mean(metrics['binary_recalls']),
        }

    avg_train_metrics = calculate_average_metrics(all_train_metrics)
    avg_test_metrics = calculate_average_metrics(all_test_metrics)
    avg_duration = np.mean(all_durations)
    avg_memory_usage = np.mean(all_memory_usages)

    # Save average statistics
    with open(os.path.join(series_results_folder, 'average_statistics.txt'), 'w') as f:
        f.write('Train Metrics:\n')
        f.write(f'Average Accuracy: {avg_train_metrics["accuracy"]:.2f}%\n')
        f.write(f'Average Precision: {avg_train_metrics["precision"]:.2f}%\n')
        f.write(f'Average Recall: {avg_train_metrics["recall"]:.2f}%\n')
        f.write(f'Average Duration: {avg_duration:.2f} seconds\n')
        f.write(f'Average Max Memory Usage: {avg_memory_usage:.2f} MB\n')
        f.write('Average Confusion Matrix:\n')
        f.write(np.array2string(avg_train_metrics["confusion_matrix"], formatter={'float_kind':lambda x: "%.2f" % x}))
        f.write('\n\n')
        f.write(f'Average Binary Accuracy: {avg_train_metrics["binary_accuracy"]:.2f}%\n')
        f.write(f'Average Binary Precision: {avg_train_metrics["binary_precision"]:.2f}%\n')
        f.write(f'Average Binary Recall: {avg_train_metrics["binary_recall"]:.2f}%\n')
        f.write('Average Binary Confusion Matrix:\n')
        f.write(np.array2string(avg_train_metrics["binary_confusion_matrix"], formatter={'float_kind':lambda x: "%.2f" % x}))
        f.write('\n\nTest Metrics:\n')
        f.write(f'Average Accuracy: {avg_test_metrics["accuracy"]:.2f}%\n')
        f.write(f'Average Precision: {avg_test_metrics["precision"]:.2f}%\n')
        f.write(f'Average Recall: {avg_test_metrics["recall"]:.2f}%\n')
        f.write(f'Average Duration: {avg_duration:.2f} seconds\n')
        f.write(f'Average Max Memory Usage: {avg_memory_usage:.2f} MB\n')
        f.write('Average Confusion Matrix:\n')
        f.write(np.array2string(avg_test_metrics["confusion_matrix"], formatter={'float_kind':lambda x: "%.2f" % x}))
        f.write('\n\n')
        f.write(f'Average Binary Accuracy: {avg_test_metrics["binary_accuracy"]:.2f}%\n')
        f.write(f'Average Binary Precision: {avg_test_metrics["binary_precision"]:.2f}%\n')
        f.write(f'Average Binary Recall: {avg_test_metrics["binary_recall"]:.2f}%\n')
        f.write('Average Binary Confusion Matrix:\n')
        f.write(np.array2string(avg_test_metrics["binary_confusion_matrix"], formatter={'float_kind':lambda x: "%.2f" % x}))

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_train_metrics['confusion_matrix'], annot=True, fmt='.2f', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Average Train Confusion Matrix')
    plt.savefig(os.path.join(series_results_folder, 'average_train_confusion_matrix.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_train_metrics['binary_confusion_matrix'], annot=True, fmt='.2f', cmap='Blues', xticklabels=['neitherFireNorSmoke', 'fire_and_smoke'], yticklabels=['neitherFireNorSmoke', 'fire_and_smoke'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Average Train Binary Confusion Matrix')
    plt.savefig(os.path.join(series_results_folder, 'average_train_binary_confusion_matrix.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_test_metrics['confusion_matrix'], annot=True, fmt='.2f', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Average Test Confusion Matrix')
    plt.savefig(os.path.join(series_results_folder, 'average_test_confusion_matrix.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_test_metrics['binary_confusion_matrix'], annot=True, fmt='.2f', cmap='Blues', xticklabels=['neitherFireNorSmoke', 'fire_and_smoke'], yticklabels=['neitherFireNorSmoke', 'fire_and_smoke'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Average Test Binary Confusion Matrix')
    plt.savefig(os.path.join(series_results_folder, 'average_test_binary_confusion_matrix.png'))
    plt.close()

    print(f'All results saved to {series_results_folder}')

if __name__ == "__main__":
    main()
