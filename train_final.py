import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# --- SETTINGS ---
DATA_DIR = "ROI_Dataset"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_system():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    class_names = full_dataset.classes

    model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'val_acc': []}

    print(f"Training... If you stop with Ctrl+C, the Matrix WILL still generate.")

    try:
        for epoch in range(EPOCHS):
            model.train()
            r_correct = 0
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, preds = torch.max(outputs, 1)
                r_correct += torch.sum(preds == labels.data)

            model.eval()
            v_correct = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    v_correct += torch.sum(preds == labels.data)

            t_acc = r_correct.double() / train_size
            v_acc = v_correct.double() / val_size
            history['train_acc'].append(t_acc.item())
            history['val_acc'].append(v_acc.item())

            print(f"Epoch {epoch+1} | Train: {t_acc:.4f} | Val: {v_acc:.4f}")
            
            # Save the latest state in case of crash
            torch.save(model.state_dict(), "model_checkpoint.pth")

    except KeyboardInterrupt:
        print("\n[STOPPED] User interrupted. Generating Matrix now...")

    # --- THIS PART RUNS EVEN IF YOU CLICK CTRL+C ---
    print("Saving Visuals...")
    
    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.legend()
    plt.savefig('accuracy_curve.png')

    # Confusion Matrix
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.savefig('confusion_matrix.png')
    print("Done! Files 'accuracy_curve.png' and 'confusion_matrix.png' are ready.")

if __name__ == "__main__":
    train_system()