# Assuming X is the matrix of gene expressions (or SNPs) and y is the ICBT treatment score:
model = Lasso(lam=1.0, lr=1.0, logistic=False)
model.fit(X, y)

# Get coefficients
beta = model.getBeta()

# Select genes with non-zero coefficients
selected_genes = np.where(beta != 0)[0]
X_selected = X[:, selected_genes]


import torch
from torch.utils.data import TensorDataset, random_split

# Convert data to TensorDataset
dataset = TensorDataset(torch.from_numpy(X_selected).float(), torch.from_numpy(y).float())

# Compute lengths of splits
train_length = int(0.2 * len(dataset))
val_length = len(dataset) - train_length

# Randomly split dataset
train_dataset, val_dataset = random_split(dataset, [train_length, val_length])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define a simple feed-forward neural network
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net(X_selected.shape[1])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate the model
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss/len(val_loader):.4f}')


def mean_absolute_error(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))

def evaluate_model(model, data_loader, criterion_mse, criterion_mae):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            total_mse += criterion_mse(outputs, batch_y).item() * batch_X.size(0)
            total_mae += criterion_mae(outputs, batch_y).item() * batch_X.size(0)
            total_samples += batch_X.size(0)
    
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    
    return avg_mse, avg_mae

# After training the model, evaluate it
criterion_mse = nn.MSELoss()
criterion_mae = mean_absolute_error

mse, mae = evaluate_model(model, val_loader, criterion_mse, criterion_mae)
print(f'Mean Squared Error on Validation Set: {mse:.4f}')
print(f'Mean Absolute Error on Validation Set: {mae:.4f}')
