import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Define the dataset class
class EyeDiseaseDataset(Dataset):
    """
    Custom Dataset for loading and preprocessing eye disease images.
    Supports the dataset presentation requirement (point 2a).
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize dataset.

        Args:
            image_paths (list): List of image file paths
            labels (list): List of corresponding labels
            transform: PyTorch transforms for data augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = ['normal', 'diabetic_retinopathy', 'cataract', 'glaucoma']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image using OpenCV
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


# Define the CNN model with biological inspiration
class BioInspiredCNN(nn.Module):
    """
    Biologically-inspired CNN mimicking human visual cortex organization.
    Used for presentation point 1d (solution description).
    """

    def __init__(self, num_classes=4):
        super(BioInspiredCNN, self).__init__()

        # V1-like layer: Low-level feature detection (edges, textures)
        self.v1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )

        # V2-like layer: Mid-level feature integration
        self.v2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )

        # V4-like layer: Higher-level feature processing
        self.v4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )

        # IT cortex-like layer: Classification
        self.it = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.v1(x)
        x = self.v2(x)
        x = self.v4(x)

        # Apply attention
        attention_bridges = self.attention(x)
        x = x * attention_bridges

        x = self.it(x)
        return x, attention_bridges


def get_data_loaders(data_dir, batch_size=32):
    """
    Prepare data loaders for training, validation, and testing.
    Supports dataset presentation (point 2a).

    Args:
        data_dir (str): Path to dataset directory
        batch_size (int): Batch size for data loaders

    Returns:
        tuple: Train, validation, and test data loaders, and dataset stats
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    class_folders = ['normal', 'diabetic_retinopathy', 'cataract', 'glaucoma']
    image_paths = []
    labels = []

    for idx, class_name in enumerate(class_folders):
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            image_paths.append(os.path.join(class_path, img_name))
            labels.append(idx)

    # Generate dataset statistics for presentation (point 1b)
    class_counts = Counter(labels)
    dataset_stats = {
        'total_images': len(image_paths),
        'class_distribution': {class_folders[i]: class_counts[i] for i in range(len(class_folders))},
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15
    }

    # Split dataset
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, stratify=labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # Create datasets
    train_dataset = EyeDiseaseDataset(train_paths, train_labels, train_transform)
    val_dataset = EyeDiseaseDataset(val_paths, val_labels, test_transform)
    test_dataset = EyeDiseaseDataset(test_paths, test_labels, test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, dataset_stats


def train_model(model, train_loader, val_loader, num_epochs=50, patience=5, device='cuda'):
    """
    Train the model with early stopping and learning rate scheduling.
    Uses SGD with momentum (not Adam) as per instructor's guidance.
    Supports presentation points 1d, 1f (parameters).

    Args:
        model: CNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs (int): Maximum number of epochs
        patience (int): Patience for early stopping
        device (str): Device to train on (cuda/cpu)

    Returns:
        tuple: Trained model and training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.1)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break

    return model, history


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the model on test data.
    Supports presentation points 1e, 1f (results and parameters).

    Args:
        model: Trained model
        test_loader: Test data loader
        device (str): Device to evaluate on

    Returns:
        tuple: Metrics and confusion matrix
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, cm


def visualize_results(history, cm, class_names):
    """
    Visualize training history and confusion matrix.
    Supports presentation point 1e (results visualization).

    Args:
        history (dict): Training history
        cm: Confusion matrix
        class_names (list): List of class names
    """
    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')


def visualize_attention(model, test_loader, device='cuda'):
    """
    Visualize attention weights for sample images.
    Supports presentation point 2d (unit data demo).

    Args:
        model: Trained model
        test_loader: Test data loader
        device (str): Device to evaluate on
    """
    model.eval()
    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images.to(device)

    with torch.no_grad():
        _, attention_weights = model(sample_images)

    attention_weights = attention_weights.cpu().numpy()

    for i in range(min(4, len(sample_images))):  # Visualize first 4 images
        plt.figure(figsize=(10, 4))

        # Original image
        plt.subplot(1, 2, 1)
        img = sample_images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        plt.imshow(img)
        plt.title(f'Original Image (Label: {test_loader.dataset.class_names[sample_labels[i]]})')

        # Attention heatmap
        plt.subplot(1, 2, 2)
        attention = attention_weights[i, 0]
        plt.imshow(attention, cmap='hot')
        plt.title('Attention Heatmap')

        plt.tight_layout()
        plt.savefig(f'attention_sample_{i}.png')


def classify_single_image(model, image_path, device='cuda'):
    """
    Classify a single retinal image using the trained model.
    Supports presentation point 2d (unit data demo).

    Args:
        model: Trained BioInspiredCNN model
        image_path (str): Path to the input image
        device (str): Device to perform inference on (cuda/cpu)

    Returns:
        tuple: Predicted class name and confidence scores
    """
    # Define transform (same as test transform)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0).to(device)

    # Set model to evaluation mode
    model.eval()
    class_names = ['normal', 'diabetic_retinopathy', 'cataract', 'glaucoma']

    with torch.no_grad():
        outputs, attention_weights = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_names[predicted.item()]
        confidence_scores = probabilities.cpu().numpy()[0]

    # Visualize attention for the single image
    attention = attention_weights.cpu().numpy()[0, 0]
    plt.figure(figsize=(10, 4))

    # Original image
    plt.subplot(1, 2, 1)
    img = image.cpu().numpy()[0].transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class} (Confidence: {confidence.item():.4f})')

    # Attention heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(attention, cmap='hot')
    plt.title('Attention Heatmap')

    plt.tight_layout()
    plt.savefig('single_image_prediction.png')

    return predicted_class, confidence_scores


def generate_dataset_stats(dataset_stats):
    """
    Generate dataset statistics for presentation (point 1b).

    Args:
        dataset_stats (dict): Dataset statistics from get_data_loaders

    Returns:
        dict: Formatted statistics for presentation
    """
    stats_str = {
        'Total Images': dataset_stats['total_images'],
        'Class Distribution': dataset_stats['class_distribution'],
        'Train Split': f"{dataset_stats['train_split'] * 100:.0f}% ({int(dataset_stats['total_images'] * dataset_stats['train_split'])} images)",
        'Validation Split': f"{dataset_stats['val_split'] * 100:.0f}% ({int(dataset_stats['total_images'] * dataset_stats['val_split'])} images)",
        'Test Split': f"{dataset_stats['test_split'] * 100:.0f}% ({int(dataset_stats['total_images'] * dataset_stats['test_split'])} images)"
    }

    # Save stats to file for presentation
    with open('dataset_stats.txt', 'w') as f:
        for key, value in stats_str.items():
            f.write(f"{key}: {value}\n")

    return stats_str


def main():
    """
    Main function to run the eye disease classification pipeline and prepare presentation materials.
    Aligns with presentation points 1a-g, 2a-d.
    """
    # Configuration
    data_dir = 'C:/Users/macie/PycharmProjects/EyeDiseaseClassification/eye_diseases_dataset'  # Update with actual path
    batch_size = 32
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data loaders and dataset stats
    train_loader, val_loader, test_loader, dataset_stats = get_data_loaders(data_dir,

                                                                            batch_size)

    # Generate dataset statistics for presentation
    stats = generate_dataset_stats(dataset_stats)
    print('\nDataset Statistics for Presentation:')
    for key, value in stats.items():
        print(f'{key}: {value}')

    # Initialize model
    model = BioInspiredCNN(num_classes=4).to(device)

    # Train model
    model, history = train_model(model, train_loader, val_loader, num_epochs, device=device)

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate model
    accuracy, precision, recall, f1, cm = evaluate_model(model, test_loader, device)
    print('\nTest Metrics (for Presentation):')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')

    # Save metrics for presentation
    with open('test_metrics.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1-score: {f1:.4f}\n')

    # Visualize results
    class_names = ['normal', 'diabetic_retinopathy', 'cataract', 'glaucoma']
    visualize_results(history, cm, class_names)
    visualize_attention(model, test_loader, device)

    # Example: Classify a single image
    example_image_path = 'C:/Users/macie/PycharmProjects/EyeDiseaseClassification/eye_diseases_dataset/glaucoma/_19_6182574.jpg'  # Update with actual image path
    try:
        predicted_class, confidence_scores = classify_single_image(model, example_image_path, device)
        print(f'\nSingle Image Classification (for Presentation Demo):')
        print(f'Predicted Class: {predicted_class}')
        print(f'Confidence Scores: {dict(zip(class_names, confidence_scores))}')
    except ValueError as e:
        print(f'Error classifying single image: {e}')


if __name__ == '__main__':
    main()