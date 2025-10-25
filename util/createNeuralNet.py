import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from csvCleaner import CSVPreprocessor

# --- 1. Custom Dataset Class (Unchanged) ---
class NumpyDataset(Dataset):
    """
    Custom Dataset to load NumPy feature and label arrays.
    Converts data to PyTorch Tensors.
    """
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        # Squeeze labels to be 1D, which CrossEntropyLoss expects
        self.labels = torch.tensor(labels.squeeze(), dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --- 2. Dynamic Neural Network Class (Slightly modified) ---
class DynamicANN(nn.Module):
    """
    A dynamically configured Artificial Neural Network.
    """
    def __init__(self, input_features, num_epochs):
        super(DynamicANN, self).__init__()
        
        self.input_features = input_features
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr = 0.001)

        self.num_epochs = num_epochs
        
        self.layer_stack = nn.Sequential(
            nn.Linear(self.input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layer_stack(x)

    def train_model(self, train_loader):
        """
        Runs one full epoch of training.
        """
        self.train()
        total_train_loss = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = self(features)
            loss = self.loss_func(outputs, labels) # labels are already 1D
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        return total_train_loss / len(train_loader)

    def test_model(self, test_loader):
        """
        Evaluates the model on the test dataset.
        """
        self.eval()
        total_test_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                outputs = self(features)
                loss = self.loss_func(outputs, labels)
                total_test_loss += loss.item()
                
                _, predicted_indices = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted_indices == labels).sum().item()
                
        avg_loss = total_test_loss / len(test_loader)
        accuracy = 100 * correct_predictions / total_samples
        return avg_loss, accuracy

def split_data(input, label, test_size = 0.2, random_state = 42):
    return train_test_split(input, label, test_size = test_size, random_state = random_state)

# --- 4. Main execution block ---
if __name__ == "__main__":
    # --- A. Data Generation ---
    processor = CSVPreprocessor("../datasets/housing.csv")
    clean_data = processor.clean()

    print(clean_data)
    input_data, label_data = processor.get_features("price")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    # --- B. Initialize and Run Workflow ---
    ann = DynamicANN(input_data, 100)
    
