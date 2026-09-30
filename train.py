import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

# Configuration
DATA_PATH = "synthetic_flicker_data.npy"
OUTPUT_MODEL_PATH = "model.pth"
BATCH_SIZE = 32
HIDDEN_DIM = 128  # Increased hidden dimension for higher model capacity
NUM_LAYERS = 3    # Increased number of LSTM layers
EPOCHS = 20       # Increased number of epochs for better learning
LEARNING_RATE = 0.0005  # Lower learning rate for more stable convergence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, 
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, 
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Add residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TimeSeriesFlickerRemovalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, kernel_size=3, dropout=0.2):
        super(TimeSeriesFlickerRemovalModel, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else hidden_dim
            out_channels = hidden_dim
            padding = (kernel_size - 1) * dilation_size // 2
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                        dilation=dilation_size, padding=padding, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Rearrange input dimensions (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Pass through TCN layers
        out = self.network(x)

        # Global average pooling over the sequence length
        pooled_out = torch.mean(out, dim=2)

        # Output layer
        return self.fc(pooled_out)

# Example use-case:
# input_dim = number of input features per time step
# hidden_dim = number of output channels for the conv layers
# output_dim = number of output features (e.g., 1 for a single output)


# Example use-case:
# input_dim = number of input features per time step
# hidden_dim = number of features in the transformer
# output_dim = number of output features (e.g., 1 for a single output)

def train_model(model, dataloader, num_epochs, learning_rate, device):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for flicker, deflicker in dataloader:
            flicker, deflicker = flicker.to(device), deflicker.to(device)
            
            optimizer.zero_grad()
            outputs = model(flicker)
            loss = criterion(outputs, deflicker)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

def test_model(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for flicker, deflicker in test_loader:
            flicker, deflicker = flicker.to(device), deflicker.to(device)
            outputs = model(flicker)
            loss = criterion(outputs, deflicker)
            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)
    print(f"Test Loss (MSE): {average_loss:.4f}")
    accuracy = 1 / (1 + average_loss)  # Example formula to convert MSE to a pseudo-accuracy
    print(f"Test Accuracy (approx): {accuracy:.4f}")
    return accuracy

def main():
    # Load data
    data = np.load(DATA_PATH, allow_pickle=True)
    flicker_data = np.array([item[0] for item in data], dtype=np.float32)
    deflicker_data = np.array([item[1] for item in data], dtype=np.float32)
    
    window_length = flicker_data.shape[1]
    input_dim = 1
    output_dim = window_length

    flicker_tensor = torch.tensor(flicker_data).unsqueeze(-1)
    deflicker_tensor = torch.tensor(deflicker_data)

    # Create dataset and split into training and testing
    dataset = TensorDataset(flicker_tensor, deflicker_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TimeSeriesFlickerRemovalModel(input_dim, HIDDEN_DIM, output_dim, NUM_LAYERS).to(DEVICE)

    # Train the model
    train_model(model, train_loader, EPOCHS, LEARNING_RATE, DEVICE)

    # Evaluate the model on test data
    accuracy = test_model(model, test_loader, DEVICE)

    # Save the model
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"Model saved to {OUTPUT_MODEL_PATH} with test accuracy of {accuracy:.4f}")

if __name__ == "__main__":
    main()