import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. Custom Dataset Class ---
# This class converts our NumPy data into a PyTorch-compatible Dataset
class NumpyDataset(Dataset):
    """
    Custom Dataset to load NumPy feature and label arrays.
    Converts data to PyTorch Tensors.
    """
    def __init__(self, features, labels):
        # Convert numpy arrays to torch tensors
        # Features are float32 (standard for model inputs)
        self.features = torch.tensor(features, dtype=torch.float32)
        
        # Labels for classification should be long (int64)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        # Returns the total number of samples
        return len(self.features)

    def __getitem__(self, idx):
        # Returns one sample (feature, label) at the given index
        return self.features[idx], self.labels[idx]

# --- 2. Dynamic Neural Network Class ---
class DynamicANN(nn.Module):
    """
    A dynamically configured Artificial Neural Network.
    
    The input_features are set to (total_columns - 1)
    The num_classes are set based on the unique values in the label column.
    """
    def __init__(self, input_features, num_classes):
        super(DynamicANN, self).__init__()
        
        print(f"Initializing DynamicANN:")
        print(f"  Input Features: {input_features}")
        print(f"  Output Classes: {num_classes}")
        
        # Define the network layers
        self.layer_stack = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # The output layer has 'num_classes' neurons
            # We don't use a Softmax layer here because
            # nn.CrossEntropyLoss (used in training) combines them.
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Defines the forward pass
        return self.layer_stack(x)

    def train_model(self, train_loader, optimizer, criterion, device):
        """
        Runs one full epoch of training.
        """
        # Set the model to training mode (enables dropout, etc.)
        self.train()
        
        total_train_loss = 0
        for features, labels in train_loader:
            # Move data to the selected device (e.g., GPU)
            features = features.to(device)
            # Labels need to be 1D and long for CrossEntropyLoss
            labels = labels.squeeze().to(device) 

            # 1. Zero the gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            outputs = self(features)
            
            # 3. Calculate loss
            loss = criterion(outputs, labels)
            
            # 4. Backward pass (backpropagation)
            loss.backward()
            
            # 5. Update weights
            optimizer.step()
            
            total_train_loss += loss.item()
            
        return total_train_loss / len(train_loader)

    def test_model(self, test_loader, criterion, device):
        """
        Evaluates the model on the test dataset.
        """
        # Set the model to evaluation mode (disables dropout, etc.)
        self.eval()
        
        total_test_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        # We don't need to calculate gradients during testing
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.squeeze().to(device)

                # Get model predictions
                outputs = self(features)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                
                # Calculate accuracy
                # torch.max returns (values, indices) along a dimension
                _, predicted_indices = torch.max(outputs.data, 1)
                
                total_samples += labels.size(0)
                correct_predictions += (predicted_indices == labels).sum().item()
                
        avg_loss = total_test_loss / len(test_loader)
        accuracy = 100 * correct_predictions / total_samples
        return avg_loss, accuracy


# --- 3. Main execution block ---
# This code only runs when the script is executed directly
if __name__ == "__main__":
    
    # --- Hyperparameters ---
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # --- A. Data Generation ---
    print("Generating random dataset...")
    np.random.seed(42) # for reproducible results
    
    # [1200.  2.  0.  0.]
    # Col 1: 1000-2000
    feature1 = np.random.uniform(1000, 2000, (1000, 1))
    # Col 2: 1-3
    feature2 = np.random.randint(1, 4, (1000, 1))
    # Col 3: 0 or 1
    feature3 = np.random.randint(0, 2, (1000, 1))
    
    # Col 4: Label (0, 1, or 2) - This will be our target
    labels = np.random.randint(0, 3, (1000, 1))
    
    # Combine features
    features = np.hstack((feature1, feature2, feature3))
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape:   {labels.shape}")

    # --- B. Get Dynamic Parameters ---
    # Input features = number of columns in the feature array
    INPUT_FEATURES = features.shape[1]
    
    # Number of classes = number of unique values in the label array
    NUM_CLASSES = len(np.unique(labels))

    # --- C. Data Splitting ---
    # Split data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Training samples:   {X_train.shape[0]}")
    print(f"  Testing samples:    {X_test.shape[0]}")

    # --- D. Create Datasets and DataLoaders ---
    # Create Dataset instances
    train_dataset = NumpyDataset(X_train, y_train)
    test_dataset = NumpyDataset(X_test, y_test)
    
    # Create DataLoader instances
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True  # Shuffle training data
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False # No need to shuffle test data
    )

    # --- E. Initialize Model and Training Components ---
    # Set device to GPU (cuda) if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Initialize the model and move it to the device
    model = DynamicANN(INPUT_FEATURES, NUM_CLASSES).to(device)
    
    # Loss function (CrossEntropy for multi-class classification)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (Adam is a good default)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- F. Training and Testing Loop ---
    print("\nStarting model training...")
    for epoch in range(EPOCHS):
        
        # Run training for one epoch
        train_loss = model.train_model(train_loader, optimizer, criterion, device)
        
        # Run testing
        test_loss, test_accuracy = model.test_model(test_loader, criterion, device)
        
        # Print epoch results
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_accuracy:.2f}%")

    print("\nTraining finished.")
