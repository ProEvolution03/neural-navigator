import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader
from model import NeuralNavigator

EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    # Make sure 'assignment_dataset' folder exists in current directory
    train_loader = get_dataloader('assignment_dataset', split='data', batch_size=BATCH_SIZE)
    model = NeuralNavigator().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print(f"Training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for images, texts, targets in train_loader:
            images, texts, targets = images.to(DEVICE), texts.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(images, texts)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "navigator_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()