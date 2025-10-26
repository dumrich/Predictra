import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset # Correctly using TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from csvCleaner import CSVCleaner
from sklearn.preprocessing import StandardScaler

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
    def __init__(self, input_features):
        super(DynamicANN, self).__init__()
        
        self.input_features = input_features
        self.loss_func = nn.MSELoss()

        self.layer_stack = nn.Sequential(
            nn.Linear(self.input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # <-- ADDED DROPOUT (p=0.5 is a good start)
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # <-- ADDED DROPOUT (p=0.5 is a good start)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),  # <-- ADDED DROPOUT (p=0.5 is a good start)
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layer_stack(x)

    def train_model(self, optimizer, train_loader, test_loader, num_epochs=100):
        """
        Runs the full training and testing loop for num_epochs.
        """
        # To store loss history for plotting
        train_losses = []
        test_losses = []

        # --- This is the main loop you were missing ---
        for epoch in range(num_epochs):
            
            # --- Training Step ---
            self.train() # Set model to training mode
            total_train_loss = 0
            
            # This loop iterates over the batches
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(features)
                
                # Ensure output and label shapes are compatible
                # For MSELoss, they should both be something like [batch_size, 1]
                loss = self.loss_func(outputs, labels) 
                
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            # Calculate average training loss for this epoch
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # --- Testing (Validation) Step ---
            # Call your test_model function to get the test loss
            avg_test_loss = self.test_model(test_loader)
            test_losses.append(avg_test_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 1:
                print(f"Epoch [{epoch+1:03d}/{num_epochs:03d}] | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Test Loss: {avg_test_loss:.4f}")
        
        print("\nTraining Finished.")
        
        # Return the loss history
        return train_losses, test_losses

    def test_model(self, test_loader):
        """
        Evaluates the model on the test dataset for one epoch.
        (This function was already correct)
        """
        self.eval() # Set model to evaluation mode
        total_test_loss = 0

        # No gradients needed for testing
        with torch.no_grad():
            for features, labels in test_loader:
                outputs = self.forward(features)
                loss = self.loss_func(outputs, labels)
                total_test_loss += loss.item()

        avg_loss = total_test_loss / len(test_loader)
        return avg_loss

def create_dataset(dataset, feature, test_size = 0.2, batch_size = 32):
    # --- A. Data Generation ---
    cleaner = CSVCleaner(dataset)
    cleaner.load_csv()
    print(f"CSV File: {cleaner.filename}")
    print(f"Columns: {cleaner.header}")

    numeric_data = cleaner.encode_data()
    print("\n=== Cleaned Numeric Data ===")
    print(numeric_data)

    print("\n=== Encoded Value Mappings ===")
    print(cleaner.encoders)

    # Automatically use lowercase 'price' as target
    X, y = cleaner.split_features_labels(feature)
    print("\n=== Features (X) Shape ===", X.shape)
    print("=== Labels (y) Shape ===", y.shape)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit and transform the data
    # Note: We fit on the *whole* dataset before splitting
    X_scaled = scaler_X.fit_transform(X)
    
    # Reshape y to (N, 1) for the scaler, then transform
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    # --- END SCALING STEP ---

    # NOW, split the SCALED data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=42
    )

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )

    # Test DataLoader
    # shuffle=False: No need to shuffle test data
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    return (train_loader, test_loader, X.shape[1], scaler_y)

def create_ann(shape): 
    # --- B. Initialize and Run Workflow ---
    ann = DynamicANN(shape)
    return ann

def split_data(input, label, test_size = 0.2, random_state = 42):
    return train_test_split(input, label, test_size = test_size, random_state = random_state)

# --- 4. Main execution block ---
if __name__ == "__main__":
    train_loader, test_loader, x_shape, scaler_y = create_dataset("../datasets/housing.csv", "price", test_size = 0.2)
    ann = create_ann(x_shape)
    # Create the optimizer
    optimizer = optim.Adam(ann.parameters(), lr=0.001, weight_decay=1e-5)
# Get just one batch of data

    ann.train_model(optimizer, train_loader, test_loader, num_epochs = 100)
