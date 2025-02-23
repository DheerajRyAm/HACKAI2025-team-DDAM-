import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    # ---------------------------
    # 1. Hyperparameters & Setup
    # ---------------------------
    DATASET_DIR = "/home/micah/Documents/Code/GitHub/HACKAI2025-team-DDAM-/backend/data/archive faces"  # <-- Update this path!
    NUM_CLASSES = 10                        # Number of gesture classes
    IMG_SIZE = (64, 64)                    # Resize images to 64x64
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    SAVE_INTERVAL = 10
    LR = 0.001                              # Learning rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------
    # 2. Data Transform & Loaders
    # ---------------------------
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # If images are RGB, use 3-channel normalization
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(DATASET_DIR, 'train'), 
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATASET_DIR, 'test'),  
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=2, pin_memory=True
    )

    # ---------------------------
    # 3. Define Model
    # ---------------------------
    class GestureCNN(nn.Module):
        def __init__(self, num_classes):
            super(GestureCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            # After 4 pooling layers on 64x64 -> (64 / 2^4 = 4) in each dimension
            self.flatten_size = 256 * (IMG_SIZE[0] // 16) * (IMG_SIZE[1] // 16)
            self.classifier = nn.Sequential(
                nn.Linear(self.flatten_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    model = GestureCNN(NUM_CLASSES).to(device)

    # ---------------------------
    # 4. Define Loss & Optimizer
    # ---------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---------------------------
    # 5. Training Loop
    # ---------------------------
    epoch_losses = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {avg_loss:.4f}")

        # Save model periodically
        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(DATASET_DIR, f'gesture_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

    # ---------------------------
    # 6. Save Final Model
    # ---------------------------
    final_model_path = os.path.join(DATASET_DIR, 'gesture_model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

    # ---------------------------
    # 7. (Optional) Evaluate on Test Set
    # ---------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Print out epoch losses
    print("\nTraining Loss per Epoch:")
    for idx, loss in enumerate(epoch_losses, 1):
        print(f"Epoch {idx}: {loss:.4f}")

# ---------------------------
# Entry point for script
# ---------------------------
if __name__ == "__main__":
    main()