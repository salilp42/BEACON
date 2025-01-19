import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from beacon import UncertaintyModel
import matplotlib.pyplot as plt

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                 transform=transform)
    test_dataset = datasets.MNIST('./data', train=False,
                                transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Create model with uncertainty quantification
    base_model = SimpleCNN()
    uncertain_model = UncertaintyModel(
        base_model=base_model,
        method='deep_ensemble',  # Try 'swag' or 'bbp' as well
        num_models=5
    )
    
    # Train model
    history = uncertain_model.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=10,
        learning_rate=1e-3
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_history.png')
    
    # Generate predictions with uncertainty
    predictions, uncertainties = uncertain_model.predict(test_loader)
    
    # Evaluate uncertainty quality
    metrics = uncertain_model.evaluate_uncertainty(test_loader)
    print("\nUncertainty Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    torch.save(uncertain_model.state_dict(), 'uncertain_model.pt')

if __name__ == '__main__':
    main()
