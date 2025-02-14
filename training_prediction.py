import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv("participant_result.csv")

# Clean categorical data
df["Collided during scenario"] = df["Collided during scenario"].astype(str).str.lower().str.strip().map({"true": 1, "false": 0})
df["Gender"] = df["Gender"].astype(str).str.lower().str.strip().map({"male": 1, "female": 0})
df["Is regular to the charity"] = df["Is regular to the charity"].astype(str).str.lower().str.strip().map({"yes": 1, "no": 0})
df["Has a driving license"] = df["Has a driving license"].astype(str).str.lower().str.strip().map({"yes": 1, "no": 0})

# Filter Scenario 2 before dropping columns
#df = df[(df["Scenario number"] == 2) | (df["Scenario number"] == 3) | (df["Scenario number"] == 4)].reset_index(drop=True)
y_collision = df["Collided during scenario"]

# Drop unnecessary columns
df = df.drop(columns=['Date', 'Participant number', "reaction time", "speed of accident", "Number of Collision", "Collided during scenario", "number of steering difference (%)",
                      "number of braking difference (%)", "average velocity difference (%)", "steering velocity difference (%)",
                      "max steering difference (%)", "max deccelration difference (%)", "max acceleration difference (%)"])

# Normalize features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert to PyTorch tensors
X_tensor = torch.tensor(df_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(y_collision.values, dtype=torch.float32).unsqueeze(1)

# Train-Test-Validation Split
X_train, X_temp, y_train, y_temp = train_test_split(X_tensor.numpy(), Y_tensor.numpy(), test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert back to tensors
X_train, X_val, X_test = map(lambda x: torch.tensor(x, dtype=torch.float32), [X_train, X_val, X_test])
y_train, y_val, y_test = map(lambda y: torch.tensor(y, dtype=torch.float32).unsqueeze(1), [y_train, y_val, y_test])

# Create DataLoaders
batch_size = 16
dataloader_train = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
dataloader_test = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Define Model
class CollisionPredict(nn.Module):
    def __init__(self, input_size):
        super(CollisionPredict, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)  # Increase neurons
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
        self.dropout = nn.Dropout(0.3)  # Add dropout to prevent overfitting
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
       
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


model = CollisionPredict(X_train.shape[1])

# Training Code remains the same...


criterion = nn.BCELoss()
        
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, Y_batch in dataloader_train:
        optimizer.zero_grad()
        outputs = model(X_batch)
        Y_batch = Y_batch.view(-1, 1)
      
        
        
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, Y_val in dataloader_val:
            outputs = model(X_val)
            Y_val = Y_val.view(-1, 1)
            loss = criterion(outputs, Y_val)
            val_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(dataloader_train):.4f}, Validation Loss: {val_loss/len(dataloader_val):.4f}")

# Evaluate on Test Set
model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in dataloader_test:
        outputs = model(batch_X)
        batch_y = batch_y.view(-1,1)

        # Apply threshold to get 0 or 1 predictions
        predictions = (outputs >= 0.5).float()  # Convert probabilities to 0 or 1
        
        
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()

        # Compute accuracy
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

# Compute final accuracy
accuracy = correct / total * 100
print(f"Final Test Loss: {test_loss/len(dataloader_test):.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from captum.attr import ShapleyValueSampling, DeepLift

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(DEVICE)

# Ensure both datasets are on the correct device
background_dataset = X_tensor.to(DEVICE)
sample_dataset = X_test.to(DEVICE)

# Ensure equal number of samples
min_samples = min(background_dataset.shape[0], sample_dataset.shape[0])
background_dataset = background_dataset[:min_samples]
sample_dataset = sample_dataset[:min_samples]

# Compute SHAP values using Shapley Value Sampling
shapley = ShapleyValueSampling(model).attribute(
    sample_dataset, target=0, baselines=background_dataset
)
shapley_df = pd.DataFrame(shapley.cpu().detach().numpy(), columns=df.columns)

# Compute SHAP values using DeepLIFT
deeplift = DeepLift(model)
deeplift_attr = deeplift.attribute(sample_dataset, target=0, baselines=background_dataset)
deeplift_df = pd.DataFrame(deeplift_attr.cpu().detach().numpy(), columns=df.columns)

# Compute mean absolute values for both methods
shapley_mean = shapley_df.abs().mean()
deeplift_mean = deeplift_df.abs().mean()

# Convert to DataFrame for visualization
shapley_mean_df = pd.DataFrame(shapley_mean, columns=["Mean SHAP Value"]).T
deeplift_mean_df = pd.DataFrame(deeplift_mean, columns=["Mean DeepLIFT Value"]).T

# Function to plot heatmap
def plot_heatmap(data, title):
    plt.figure(figsize=(12, 2))
    sns.heatmap(data, cmap="coolwarm", annot=True, fmt=".4f", linewidths=0.5)
    plt.title(title)
    plt.xlabel("Features")
    plt.show()

# Plot heatmaps
plot_heatmap(shapley_mean_df, "Feature Importance Heatmap (Shapley Value Sampling)")
plot_heatmap(deeplift_mean_df, "Feature Importance Heatmap (DeepLIFT)")

