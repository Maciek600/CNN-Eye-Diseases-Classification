import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# Define the CNN model (same as in your original code)
class BioInspiredCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BioInspiredCNN, self).__init__()
        self.v1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )
        self.v2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.v4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        self.it = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.v1(x)
        x = self.v2(x)
        x = self.v4(x)
        attention_bridges = self.attention(x)
        x = x * attention_bridges
        x = self.it(x)
        return x, attention_bridges

# Function to classify a single image (adapted for GUI)
def classify_single_image(model, image_path, device='cuda'):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Nie można wczytać obrazu z {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)

    image = image.unsqueeze(0).to(device)

    model.eval()
    class_names = ['Normal', 'Diabetic retinopathy', 'Cataract', 'Glaucoma']
    with torch.no_grad():
        outputs, attention_weights = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_names[predicted.item()]
        confidence_scores = probabilities.cpu().numpy()[0]

    attention = attention_weights.cpu().numpy()[0, 0]
    plt.figure(figsize=(5, 5))
    plt.imshow(attention, cmap='hot')
    plt.axis('off')
    plt.savefig('temp_attention.png', bbox_inches='tight')
    plt.close()

    return predicted_class, confidence_scores

# GUI Application
class EyeDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye diseases classification")
        self.root.geometry("800x600")

        # Load the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BioInspiredCNN(num_classes=4).to(self.device)
        try:
            self.model.load_state_dict(torch.load('best_model.pth', map_location=self.device))
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można załadować modelu: {e}")
            self.root.destroy()
            return

        # GUI elements
        self.label = tk.Label(root, text="Select an eye image for classification", font=("Arial", 14))
        self.label.pack(pady=10)

        self.upload_btn = tk.Button(root, text="Upload a photo", command=self.upload_image, font=("Arial", 12))
        self.upload_btn.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 12), wraplength=700)
        self.result_label.pack(pady=10)

        self.image_frame = tk.Frame(root)
        self.image_frame.pack(pady=10)

        self.original_image_label = tk.Label(self.image_frame)
        self.original_image_label.pack(side=tk.LEFT, padx=10)

        self.attention_image_label = tk.Label(self.image_frame)
        self.attention_image_label.pack(side=tk.LEFT, padx=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        try:
            predicted_class, confidence_scores = classify_single_image(self.model, file_path, self.device)
            class_names = ['Normal', 'Diabetic retinopathy', 'Cataract', 'Glaucoma']
            confidence_text = "\n".join([f"{name}: {score:.4f}" for name, score in zip(class_names, confidence_scores)])
            self.result_label.config(text=f"Expected disease: {predicted_class}\nAccuracy:\n{confidence_text}")

            # Display original image
            img = Image.open(file_path)
            img = img.resize((300, 300), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.original_image_label.config(image=img_tk)
            self.original_image_label.image = img_tk

            # Display attention heatmap
            attention_img = Image.open('temp_attention.png')
            attention_img = attention_img.resize((300, 300), Image.Resampling.LANCZOS)
            attention_img_tk = ImageTk.PhotoImage(attention_img)
            self.attention_image_label.config(image=attention_img_tk)
            self.attention_image_label.image = attention_img_tk

            # Clean up temporary file
            if os.path.exists('temp_attention.png'):
                os.remove('temp_attention.png')

        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można przetworzyć obrazu: {e}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = EyeDiseaseApp(root)
    root.mainloop()