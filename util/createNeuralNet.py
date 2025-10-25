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
    def __init__(self, input_features, num_classes):
        super(DynamicANN, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        
        self.layer_stack = nn.Sequential(
            nn.Linear(self.input_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes)
        )

    def forward(self, x):
        return self.layer_stack(x)

    def train_model(self, train_loader, optimizer, criterion, device):
        """
        Runs one full epoch of training.
        """
        self.train()
        total_train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = self(features)
            loss = criterion(outputs, labels) # labels are already 1D
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        return total_train_loss / len(train_loader)

    def test_model(self, test_loader, criterion, device):
        """
        Evaluates the model on the test dataset.
        """
        self.eval()
        total_test_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)

                outputs = self(features)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()
                
                _, predicted_indices = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted_indices == labels).sum().item()
                
        avg_loss = total_test_loss / len(test_loader)
        accuracy = 100 * correct_predictions / total_samples
        return avg_loss, accuracy

# --- 3. NEW: Workflow Manager Class ---
class ANNWorkflow:
    """
    Manages the entire ANN training and evaluation workflow,
    holding configuration like epochs and test_size.
    """
    def __init__(self, epochs, test_size, batch_size, learning_rate):
        # Store configuration as instance variables
        self.epochs = epochs
        self.test_size = test_size  # The train-test split percentage (e.g., 0.2)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Workflow initialized. Using device: {self.device}")

    def prepare_data(self, full_numpy_array, label_column_index):
        """
        Takes a full numpy array and a 'key' (column index) for the label.
        Separates the array into features (everything else) and labels.
        """
        print(f"Preparing data. Using column {label_column_index} as the label.")
        
        # Ensure label_column_index is valid
        if label_column_index < 0: # Handle negative indexing
            label_column_index = full_numpy_array.shape[1] + label_column_index
            
        if not 0 <= label_column_index < full_numpy_array.shape[1]:
            raise ValueError("label_column_index is out of bounds.")
            
        # Select the label column
        # Use reshape to ensure labels are (n_samples, 1)
        labels = full_numpy_array[:, label_column_index].reshape(-1, 1)
        
        # Select all columns *except* the label column
        features = np.delete(full_numpy_array, label_column_index, axis=1)
        
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape:   {labels.shape}")
        
        return features, labels

    def run(self, full_numpy_array, label_column_index):
        """
        Executes the full workflow:
        1. Prepare data
        2. Split data
        3. Create DataLoaders
        4. Initialize model
        5. Run training and evaluation loop
        """
        
        # --- 1. Prepare Data ---
        features, labels = self.prepare_data(full_numpy_array, label_column_index)
        
        # --- 2. Get Dynamic Parameters ---
        input_features = features.shape[1]
        num_classes = len(np.unique(labels))
        
        # --- 3. Data Splitting ---
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, 
            test_size=self.test_size,  # Use instance variable
            random_state=42
        )
        print(f"\nData split ({int((1-self.test_size)*100)}% / {int(self.test_size*100)}%):")
        print(f"  Training samples:   {X_train.shape[0]}")
        print(f"  Testing samples:    {X_test.shape[0]}")

        # --- 4. Create Datasets and DataLoaders ---
        train_dataset = NumpyDataset(X_train, y_train)
        test_dataset = NumpyDataset(X_test, y_test)
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size, # Use instance variable
            shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size, # Use instance variable
            shuffle=False
        )
        
        # --- 5. Initialize Model and Training ---
        print(f"\nInitializing DynamicANN:")
        print(f"  Input Features: {input_features}")
        print(f"  Output Classes: {num_classes}")
        
        self.model = DynamicANN(input_features, num_classes).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # --- 6. Training and Testing Loop ---
        print("\nStarting model training...")
        for epoch in range(self.epochs): # Use instance variable
            
            train_loss = self.model.train_model(train_loader, optimizer, criterion, self.device)
            test_loss, test_accuracy = self.model.test_model(test_loader, criterion, self.device)
            
            print(f"Epoch [{epoch+1:02d}/{self.epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_accuracy:.2f}%")

        print("\nTraining finished.")

# --- 4. Main execution block ---
if __name__ == "__main__":
    
    # --- A. Data Generation ---
    processor = CSVPreprocessor("../datasets/housing.csv")
    clean_data = processor.clean()
    print(clean_data)
    
    # --- B. Initialize and Run Workflow ---
    
    # Configure the workflow with instance variables
    workflow = ANNWorkflow(
        epochs=40, 
        test_size=0.2,   # 20% test split
        batch_size=32, 
        learning_rate=0.001
    )
    
    # Run the workflow.
    # Tell it that our label is in column index 2
    workflow.run(
        full_numpy_array=clean_data, 
        label_column_index=2
    )
