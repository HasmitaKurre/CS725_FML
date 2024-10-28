# -*- coding: utf-8 -*-
"""
#LogisticRegression
"""

import numpy as np
import random
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

s=1337
np.random.seed(s)
random.seed(s)
tf.random.set_seed(s)

# Load the dataset
data_train = pd.read_csv('/content/train.csv')
data_test = pd.read_csv('/content/test.csv')

# Extract features and labels
X_train = data_train.iloc[:, 1:-1].values  # Features (excluding ID and label)
y_train = data_train['label'].values       # Labels

# Extract features from test data (assuming 'label' column not available in test set)
X_test = data_test.iloc[:, 1:].values  # Features (excluding ID)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=s)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)  # Standardize the test set

# Initialize the LogisticRegression model
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=s)

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = log_reg.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

# Make predictions on the test set
y_test_pred = log_reg.predict(X_test)

data = {
  'ID': data_test['ID'],
  'label': y_test_pred,
}

#data['label']= y_test_pred

df = pd.DataFrame(data)
df.to_csv('kaggle_logistic.csv', index=False)

df

"""#ANN"""

import numpy as np
import random
import pandas as pd
import tensorflow as tf

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Set random seeds for reproducibility
s = 1337
np.random.seed(s)
random.seed(s)
tf.random.set_seed(s)

# Load the dataset
data = pd.read_csv('/content/train.csv')
X = data.iloc[:, 1:-1].values  # Features (excluding ID and label)
y = data['label'].values       # Labels

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=s)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)  # Input 64 features, output 128 neurons
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 100)  # Output layer (adjust for number of classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = NeuralNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
batch_size = 32

def train_model():
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, X_train_tensor.size(0), batch_size):
            inputs = X_train_tensor[i:i+batch_size]
            labels = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, val_preds = torch.max(val_outputs, 1)
            val_accuracy = accuracy_score(y_val_tensor.numpy(), val_preds.numpy())
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}')

# Call the training function
train_model()

# Load the test dataset
test_data = pd.read_csv('/content/test.csv')
X_test = test_data.iloc[:, 1:].values  # Extract features from test data (assuming 'ID' column exists)

# Standardize the test data
X_test = scaler.transform(X_test)  # Standardize the test set
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Evaluate the model on the test dataset
def evaluate_test():
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, test_preds = torch.max(test_outputs, 1)  # Predicted labels
    return test_preds

# Call the function to get test predictions
test_predictions = evaluate_test()

# Create a DataFrame with 'ID' from the test dataset and the predicted 'label'
data = {
    'ID': test_data['ID'],              # Assuming the test data contains an 'ID' column
    'label': test_predictions.cpu().numpy()  # Convert predictions to numpy array
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('kaggle_ann.csv', index=False)

# Display the DataFrame
print(df)

"""#MLPClassifier"""

import numpy as np
import random
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Set random seeds for reproducibility
s = 1337
np.random.seed(s)
random.seed(s)
tf.random.set_seed(s)

# Load the dataset
data = pd.read_csv('/content/train.csv')

# Extract features and labels
X = data.iloc[:, 1:-1].values  # Features (excluding ID and label)
y = data['label'].values        # Labels

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=s)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Initialize the MLPClassifier (Multi-layer Perceptron)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=s)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = mlp.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

# Load the test dataset
test_data = pd.read_csv('/content/test.csv')  # Load test data
X_test = test_data.iloc[:, 1:].values  # Extract features from test data (assuming 'ID' column exists)

# Standardize the test data
X_test = scaler.transform(X_test)  # Standardize the test set

# Make predictions on the test set
test_predictions = mlp.predict(X_test)  # Predict labels for the test set

# Create a DataFrame with 'ID' from the test dataset and the predicted 'label'
data = {
    'ID': test_data['ID'],               # Assuming the test data contains an 'ID' column
    'label': test_predictions              # Predictions from the test set
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('kaggle_mlp.csv', index=False)

# Display the DataFrame
print(df)

"""#LSTM"""

import numpy as np
import random
import tensorflow as tf
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Seed setting
s = 1337
np.random.seed(s)
random.seed(s)
tf.random.set_seed(s)

# Load the training dataset
train_data = pd.read_csv('/content/train.csv')
X = train_data.iloc[:, 1:-1].values  # Features (excluding ID and label)
y = train_data['label'].values       # Labels

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=s)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to PyTorch tensors (train/val)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)      # Add sequence dimension
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Load and preprocess the test dataset
test_data = pd.read_csv('/content/test.csv')
X_test = test_data.iloc[:, 1:].values  # Features (excluding ID)
X_test = scaler.transform(X_test)  # Use the same scaler as training data

# Convert to PyTorch tensors (test set)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

# RNN model using LSTM
class RNNClassifier(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, output_size=100, num_layers=1):
        super(RNNClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Initialize the model, loss function, and optimizer
model = RNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
batch_size = 64

def train_model():
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, X_train_tensor.size(0), batch_size):
            inputs = X_train_tensor[i:i+batch_size]
            labels = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, val_preds = torch.max(val_outputs, 1)
            val_accuracy = accuracy_score(y_val_tensor, val_preds)
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}')

# Call the training function
train_model()

# Evaluate the model on the test dataset
def evaluate_test():
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, test_preds = torch.max(test_outputs, 1)  # Predicted labels
    return test_preds

# Call the function to get test predictions
test_predictions = evaluate_test()

# Create a DataFrame with 'ID' from the test dataset and the predicted 'label'
data = {
    'ID': test_data['ID'],          # Assuming the test data contains an 'ID' column
    'label': test_predictions.cpu().numpy()  # Convert predictions to numpy array
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('kaggle_lstm.csv', index=False)

# Display the DataFrame
print(df)

"""#GRU"""

import numpy as np
import random
import tensorflow as tf
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Seed setting
s = 1337
np.random.seed(s)
random.seed(s)
tf.random.set_seed(s)

# Load the training dataset
train_data = pd.read_csv('/content/train.csv')
X = train_data.iloc[:, 1:-1].values  # Features (excluding ID and label)
y = train_data['label'].values       # Labels

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=s)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to PyTorch tensors (train/val)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)      # Add sequence dimension
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Load and preprocess the test dataset
test_data = pd.read_csv('/content/test.csv')
X_test = test_data.iloc[:, 1:].values  # Features (excluding ID)
X_test = scaler.transform(X_test)  # Use the same scaler as training data

# Convert to PyTorch tensors (test set)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

# GRU model
class GRUClassifier(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, output_size=100, num_layers=1):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]  # Take the output from the last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Initialize the model, loss function, and optimizer
model = GRUClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
batch_size = 32

def train_model():
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, X_train_tensor.size(0), batch_size):
            inputs = X_train_tensor[i:i+batch_size]
            labels = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, val_preds = torch.max(val_outputs, 1)
            val_accuracy = accuracy_score(y_val_tensor, val_preds)
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}')

# Call the training function
train_model()

# Evaluate the model on the test dataset
def evaluate_test():
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, test_preds = torch.max(test_outputs, 1)  # Predicted labels
    return test_preds

# Call the function to get test predictions
test_predictions = evaluate_test()

# Create a DataFrame with 'ID' from the test dataset and the predicted 'label'
data = {
    'ID': test_data['ID'],          # Assuming the test data contains an 'ID' column
    'label': test_predictions.cpu().numpy()  # Convert predictions to numpy array
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('kaggle_gru.csv', index=False)

# Display the DataFrame
print(df)

import numpy as np
import random
import tensorflow as tf
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Seed setting
s = 1337
np.random.seed(s)
random.seed(s)
tf.random.set_seed(s)

# Load the training dataset
train_data = pd.read_csv('/content/train.csv')
X = train_data.iloc[:, 1:-1].values  # Features (excluding ID and label)
y = train_data['label'].values       # Labels

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=s)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to PyTorch tensors (train/val)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)      # Add sequence dimension
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Load and preprocess the test dataset
test_data = pd.read_csv('/content/test.csv')
X_test = test_data.iloc[:, 1:].values  # Features (excluding ID)
X_test = scaler.transform(X_test)  # Use the same scaler as training data

# Convert to PyTorch tensors (test set)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

# GRU model with increased complexity and dropout
class GRUClassifier(nn.Module):
    def __init__(self, input_size=X_train.shape[1], hidden_size=256, output_size=len(np.unique(y)), num_layers=2):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)  # Increased dropout

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]  # Take the output from the last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Initialize the model, loss function, and optimizer
model = GRUClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

# Training loop with early stopping
num_epochs = 100
batch_size = 128
best_val_accuracy = 0.0
patience = 5
patience_counter = 0

def train_model():
    global best_val_accuracy, patience_counter
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, X_train_tensor.size(0), batch_size):
            inputs = X_train_tensor[i:i+batch_size]
            labels = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, val_preds = torch.max(val_outputs, 1)
            val_accuracy = accuracy_score(y_val_tensor, val_preds)
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}')

            # Learning rate scheduling
            scheduler.step(loss)

            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0  # Reset counter if validation accuracy improves
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

# Call the training function
train_model()

# Evaluate the model on the test dataset
def evaluate_test():
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, test_preds = torch.max(test_outputs, 1)  # Predicted labels
    return test_preds

# Call the function to get test predictions
test_predictions = evaluate_test()

# Create a DataFrame with 'ID' from the test dataset and the predicted 'label'
data = {
    'ID': test_data['ID'],          # Assuming the test data contains an 'ID' column
    'label': test_predictions.cpu().numpy()  # Convert predictions to numpy array
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('kaggle_gru.csv', index=False)

# Display the DataFrame
print(df)