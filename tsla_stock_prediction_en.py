"""
TSLA Stock Price Prediction Project
Using LSTM + Transformer Hybrid Model

This module implements a deep learning approach for stock price prediction,
combining Long Short-Term Memory (LSTM) networks with Transformer architecture
to capture both sequential dependencies and attention-based patterns in
financial time series data.

Author: LSTM-Transformer Stock Prediction Project
Date: January 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

# Create output directory for saving results
os.makedirs('output', exist_ok=True)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ==============================================================================
print("=" * 60)
print("1. Data Loading and Preprocessing")
print("=" * 60)

# Load the TSLA stock data from CSV file
df = pd.read_csv('TSLA.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"\nDataset shape: {df.shape}")
print(f"Time range: {df['Date'].min()} to {df['Date'].max()}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDescriptive statistics:")
print(df.describe())

# Check for missing values
print(f"\nMissing values count:")
print(df.isnull().sum())


def add_technical_indicators(df):
    """
    Add technical analysis indicators to the dataframe.
    
    Technical indicators help capture market trends, momentum, and volatility
    patterns that may be useful for price prediction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLCV (Open, High, Low, Close, Volume) data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional technical indicator columns
    """
    # Moving Averages (MA)
    # Short-term, medium-term, and long-term trend indicators
    df['MA5'] = df['Close'].rolling(window=5).mean()    # 5-day MA
    df['MA10'] = df['Close'].rolling(window=10).mean()  # 10-day MA
    df['MA20'] = df['Close'].rolling(window=20).mean()  # 20-day MA
    
    # Price Change Rate (Daily Returns)
    df['Price_Change'] = df['Close'].pct_change()
    
    # Volatility (20-day rolling standard deviation of returns)
    df['Volatility'] = df['Price_Change'].rolling(window=20).std()
    
    # Relative Strength Index (RSI)
    # RSI measures the magnitude of recent price changes to evaluate
    # overbought or oversold conditions
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    # MACD is a trend-following momentum indicator
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()  # Fast EMA
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()  # Slow EMA
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    # Bollinger Bands measure volatility and provide relative price levels
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    return df


# Apply feature engineering
df = add_technical_indicators(df)
df = df.dropna().reset_index(drop=True)  # Remove rows with NaN values

print(f"\nDataset shape after adding technical indicators: {df.shape}")
print(f"New features: MA5, MA10, MA20, Price_Change, Volatility, RSI, MACD, Signal_Line, BB_upper, BB_lower")

# ==============================================================================
# SECTION 2: DATA VISUALIZATION
# ==============================================================================
print("\n" + "=" * 60)
print("2. Data Visualization")
print("=" * 60)

# Figure 1: Stock Price Overview with Technical Indicators
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Subplot 1: Close price with Moving Averages and Bollinger Bands
axes[0].plot(df['Date'], df['Close'], label='Close Price', linewidth=1.5, color='#2E86AB')
axes[0].plot(df['Date'], df['MA5'], label='MA5', linewidth=1, alpha=0.8, color='#F18F01')
axes[0].plot(df['Date'], df['MA20'], label='MA20', linewidth=1, alpha=0.8, color='#C73E1D')
axes[0].fill_between(df['Date'], df['BB_lower'], df['BB_upper'], alpha=0.2, color='gray', label='Bollinger Bands')
axes[0].set_title('TSLA Stock Price with Moving Averages & Bollinger Bands', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price (USD)', fontsize=12)
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Subplot 2: Trading Volume
axes[1].bar(df['Date'], df['Volume'], color='#3A86FF', alpha=0.7, width=2)
axes[1].set_title('Trading Volume', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Volume', fontsize=12)
axes[1].grid(True, alpha=0.3)

# Subplot 3: RSI Indicator with overbought/oversold thresholds
axes[2].plot(df['Date'], df['RSI'], color='#8338EC', linewidth=1.5)
axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
axes[2].fill_between(df['Date'], 30, 70, alpha=0.1, color='green')
axes[2].set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold')
axes[2].set_ylabel('RSI', fontsize=12)
axes[2].set_xlabel('Date', fontsize=12)
axes[2].legend(loc='upper left')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/01_stock_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/01_stock_overview.png")

# Figure 2: Data Distribution and Correlation Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Close Price Distribution
axes[0, 0].hist(df['Close'], bins=50, color='#2E86AB', edgecolor='white', alpha=0.8)
axes[0, 0].set_title('Close Price Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Price (USD)')
axes[0, 0].set_ylabel('Frequency')

# Subplot 2: Daily Returns Distribution
axes[0, 1].hist(df['Price_Change'].dropna(), bins=50, color='#06D6A0', edgecolor='white', alpha=0.8)
axes[0, 1].set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Daily Return')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)

# Subplot 3: OHLC Chart (simplified candlestick representation)
sample_df = df.tail(60)
colors = ['#06D6A0' if c >= o else '#EF476F' for o, c in zip(sample_df['Open'], sample_df['Close'])]
axes[1, 0].bar(range(len(sample_df)), sample_df['Close'] - sample_df['Open'], 
               bottom=sample_df['Open'], color=colors, width=0.8)
axes[1, 0].vlines(range(len(sample_df)), sample_df['Low'], sample_df['High'], color='gray', linewidth=0.5)
axes[1, 0].set_title('OHLC Chart (Last 60 Days)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Day Index')
axes[1, 0].set_ylabel('Price (USD)')

# Subplot 4: Feature Correlation Heatmap
corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD']
corr_matrix = df[corr_cols].corr()
im = axes[1, 1].imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto')
axes[1, 1].set_xticks(range(len(corr_cols)))
axes[1, 1].set_yticks(range(len(corr_cols)))
axes[1, 1].set_xticklabels(corr_cols, rotation=45, ha='right', fontsize=9)
axes[1, 1].set_yticklabels(corr_cols, fontsize=9)
axes[1, 1].set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=axes[1, 1], shrink=0.8)

# Add correlation coefficient values to heatmap
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=7)

plt.tight_layout()
plt.savefig('output/02_data_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/02_data_analysis.png")

# Figure 3: MACD Analysis
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Subplot 1: Close Price
axes[0].plot(df['Date'], df['Close'], label='Close Price', color='#2E86AB', linewidth=1.5)
axes[0].set_title('TSLA Close Price', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Price (USD)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Subplot 2: MACD Indicator
axes[1].plot(df['Date'], df['MACD'], label='MACD', color='#3A86FF', linewidth=1.5)
axes[1].plot(df['Date'], df['Signal_Line'], label='Signal Line', color='#FF006E', linewidth=1.5)
axes[1].bar(df['Date'], df['MACD'] - df['Signal_Line'], 
            color=['#06D6A0' if x >= 0 else '#EF476F' for x in df['MACD'] - df['Signal_Line']], 
            alpha=0.5, width=2)
axes[1].set_title('MACD Indicator', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('MACD')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/03_macd_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/03_macd_analysis.png")

# ==============================================================================
# SECTION 3: DATA PREPROCESSING FOR MODEL
# ==============================================================================
print("\n" + "=" * 60)
print("3. Data Preprocessing for Model")
print("=" * 60)

# Define feature columns for model input
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'MA20', 
                   'RSI', 'MACD', 'Signal_Line']
target_column = 'Close'

# Initialize scalers for normalization
# Using Min-Max scaling to normalize features to [0, 1] range
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Fit and transform the data
scaled_features = scaler_features.fit_transform(df[feature_columns])
scaled_target = scaler_target.fit_transform(df[[target_column]])

print(f"Number of features: {len(feature_columns)}")
print(f"Feature list: {feature_columns}")


def create_sequences(features, target, seq_length):
    """
    Create sequences for time series prediction using sliding window approach.
    
    This function converts the time series data into supervised learning format
    where each sample consists of `seq_length` consecutive time steps as input
    and the next time step's target value as output.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature matrix of shape (n_samples, n_features)
    target : numpy.ndarray
        Target vector of shape (n_samples, 1)
    seq_length : int
        Number of time steps to use as input sequence
        
    Returns:
    --------
    X : numpy.ndarray
        Input sequences of shape (n_sequences, seq_length, n_features)
    y : numpy.ndarray
        Target values of shape (n_sequences, 1)
    """
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)


# Set sequence length (lookback window)
SEQ_LENGTH = 30  # Use past 30 days to predict next day

# Create sequences
X, y = create_sequences(scaled_features, scaled_target, SEQ_LENGTH)

print(f"\nSequence length: {SEQ_LENGTH}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split data into training, validation, and test sets
# Using chronological split to avoid data leakage
train_size = int(len(X) * 0.7)   # 70% for training
val_size = int(len(X) * 0.15)    # 15% for validation

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"\nTraining set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Create DataLoader objects for batch processing
BATCH_SIZE = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==============================================================================
# SECTION 4: MODEL DEFINITION (LSTM + TRANSFORMER)
# ==============================================================================
print("\n" + "=" * 60)
print("4. Model Definition (LSTM + Transformer)")
print("=" * 60)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module for Transformer.
    
    Since Transformers don't have recurrence or convolution, we need to inject
    information about the relative or absolute position of tokens in the sequence.
    This implementation uses sinusoidal positional encodings.
    
    Parameters:
    -----------
    d_model : int
        The dimension of the model (embedding dimension)
    max_len : int
        Maximum sequence length for positional encoding
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the positional encodings using sin and cos functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input tensor."""
        return x + self.pe[:x.size(0), :]


class LSTMTransformerModel(nn.Module):
    """
    Hybrid LSTM-Transformer Model for Time Series Prediction.
    
    This model combines the strengths of both LSTM and Transformer architectures:
    - LSTM: Captures sequential dependencies and temporal patterns
    - Transformer: Learns complex attention-based relationships between time steps
    
    Architecture:
    1. Input Projection: Linear layer to project input features to hidden dimension
    2. Bidirectional LSTM: Captures temporal dependencies from both directions
    3. Positional Encoding: Adds position information for Transformer
    4. Transformer Encoder: Self-attention mechanism for learning relationships
    5. Output MLP: Multi-layer perceptron for final prediction
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    hidden_dim : int
        Hidden dimension for LSTM and Transformer
    lstm_layers : int
        Number of LSTM layers
    nhead : int
        Number of attention heads in Transformer
    transformer_layers : int
        Number of Transformer encoder layers
    dropout : float
        Dropout rate for regularization
    """
    def __init__(self, input_dim, hidden_dim=128, lstm_layers=2, 
                 nhead=4, transformer_layers=2, dropout=0.2):
        super(LSTMTransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection layer: project input features to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Bidirectional LSTM layer
        # Bidirectional allows the model to capture patterns from both directions
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # LSTM output projection (2*hidden_dim -> hidden_dim due to bidirectional)
        self.lstm_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Positional encoding for Transformer
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Output layers (MLP)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Output predictions of shape (batch_size, 1)
        """
        # Step 1: Input projection
        x = self.input_projection(x)  # (batch, seq_len, hidden_dim)
        
        # Step 2: LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        lstm_out = self.lstm_projection(lstm_out)  # (batch, seq_len, hidden_dim)
        
        # Step 3: Positional encoding
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch, hidden_dim)
        lstm_out = self.pos_encoder(lstm_out)
        lstm_out = lstm_out.transpose(0, 1)  # (batch, seq_len, hidden_dim)
        
        # Step 4: Transformer encoding
        transformer_out = self.transformer_encoder(lstm_out)  # (batch, seq_len, hidden_dim)
        
        # Step 5: Take the last time step output
        out = transformer_out[:, -1, :]  # (batch, hidden_dim)
        
        # Step 6: Output MLP
        out = self.fc_layers(out)  # (batch, 1)
        
        return out


# Initialize model and move to appropriate device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = LSTMTransformerModel(
    input_dim=len(feature_columns),
    hidden_dim=128,
    lstm_layers=2,
    nhead=4,
    transformer_layers=2,
    dropout=0.2
).to(device)

# Print model architecture
print("\nModel Architecture:")
print(model)

# Calculate and print parameter counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ==============================================================================
# SECTION 5: MODEL TRAINING
# ==============================================================================
print("\n" + "=" * 60)
print("5. Model Training")
print("=" * 60)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler: reduce LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Parameters:
    -----------
    model : nn.Module
        The neural network model
    loader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimization algorithm
    device : torch.device
        Device to run computations on
        
    Returns:
    --------
    float
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on a dataset.
    
    Parameters:
    -----------
    model : nn.Module
        The neural network model
    loader : DataLoader
        Data loader for evaluation
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to run computations on
        
    Returns:
    --------
    tuple
        (average_loss, predictions_array, actuals_array)
    """
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    
    return total_loss / len(loader), np.array(predictions), np.array(actuals)


# Training loop configuration
EPOCHS = 100
best_val_loss = float('inf')
train_losses = []
val_losses = []
patience_counter = 0
early_stop_patience = 20

print(f"\nStarting training (max {EPOCHS} epochs)...")

for epoch in tqdm(range(EPOCHS), desc="Training"):
    # Train for one epoch
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Evaluate on validation set
    val_loss, _, _ = evaluate(model, val_loader, criterion, device)
    
    # Record losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Update learning rate based on validation loss
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'output/best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"\nEpoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Early stopping check
    if patience_counter >= early_stop_patience:
        print(f"\nEarly stopping: Validation loss hasn't improved for {early_stop_patience} epochs")
        break

print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")

# Plot training curves
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train_losses, label='Training Loss', color='#2E86AB', linewidth=2)
ax.plot(val_losses, label='Validation Loss', color='#EF476F', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss (MSE)', fontsize=12)
ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/04_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/04_training_curves.png")

# ==============================================================================
# SECTION 6: MODEL EVALUATION AND RESULTS
# ==============================================================================
print("\n" + "=" * 60)
print("6. Model Evaluation and Results")
print("=" * 60)

# Load best model weights
model.load_state_dict(torch.load('output/best_model.pth'))

# Evaluate on test set
test_loss, test_predictions, test_actuals = evaluate(model, test_loader, criterion, device)

# Inverse transform predictions and actuals to original scale
test_predictions_inv = scaler_target.inverse_transform(test_predictions.reshape(-1, 1))
test_actuals_inv = scaler_target.inverse_transform(test_actuals.reshape(-1, 1))

# Calculate evaluation metrics
mse = mean_squared_error(test_actuals_inv, test_predictions_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_actuals_inv, test_predictions_inv)
r2 = r2_score(test_actuals_inv, test_predictions_inv)
mape = np.mean(np.abs((test_actuals_inv - test_predictions_inv) / test_actuals_inv)) * 100

print(f"\nTest Set Evaluation Metrics:")
print(f"{'='*40}")
print(f"MSE  (Mean Squared Error):        {mse:.4f}")
print(f"RMSE (Root Mean Squared Error):   {rmse:.4f}")
print(f"MAE  (Mean Absolute Error):       {mae:.4f}")
print(f"R²   (Coefficient of Determination): {r2:.4f}")
print(f"MAPE (Mean Absolute % Error):     {mape:.2f}%")
print(f"{'='*40}")

# Save evaluation metrics to CSV
metrics = {
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae,
    'R2': r2,
    'MAPE': mape
}
pd.DataFrame([metrics]).to_csv('output/evaluation_metrics.csv', index=False)
print("\nSaved evaluation metrics: output/evaluation_metrics.csv")

# Create prediction results visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Actual vs Predicted prices (time series)
test_dates = df['Date'].iloc[train_size+val_size+SEQ_LENGTH:].reset_index(drop=True)
axes[0, 0].plot(test_dates, test_actuals_inv, label='Actual Price', color='#2E86AB', linewidth=2)
axes[0, 0].plot(test_dates, test_predictions_inv, label='Predicted Price', color='#EF476F', linewidth=2, linestyle='--')
axes[0, 0].set_title('Test Set: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price (USD)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Subplot 2: Prediction Error Distribution
errors = test_predictions_inv.flatten() - test_actuals_inv.flatten()
axes[0, 1].hist(errors, bins=30, color='#3A86FF', edgecolor='white', alpha=0.8)
axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean Error: {errors.mean():.2f}')
axes[0, 1].set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Prediction Error (USD)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

# Subplot 3: Scatter plot (Actual vs Predicted)
axes[1, 0].scatter(test_actuals_inv, test_predictions_inv, alpha=0.6, color='#8338EC', s=50)
min_val = min(test_actuals_inv.min(), test_predictions_inv.min())
max_val = max(test_actuals_inv.max(), test_predictions_inv.max())
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[1, 0].set_title(f'Actual vs Predicted (R² = {r2:.4f})', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Actual Price (USD)')
axes[1, 0].set_ylabel('Predicted Price (USD)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Evaluation Metrics Bar Chart
metrics_names = ['RMSE', 'MAE', 'MAPE(%)']
metrics_values = [rmse, mae, mape]
colors = ['#2E86AB', '#06D6A0', '#F18F01']
bars = axes[1, 1].bar(metrics_names, metrics_values, color=colors, edgecolor='white', linewidth=2)
axes[1, 1].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Value')
for bar, val in zip(bars, metrics_values):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('output/05_prediction_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/05_prediction_results.png")

# Generate predictions for full dataset
print("\nGenerating predictions for full dataset...")
model.eval()
all_predictions = []

with torch.no_grad():
    for i in range(len(X)):
        x_input = torch.FloatTensor(X[i:i+1]).to(device)
        pred = model(x_input)
        all_predictions.append(pred.cpu().numpy()[0, 0])

all_predictions = np.array(all_predictions)
all_predictions_inv = scaler_target.inverse_transform(all_predictions.reshape(-1, 1))
all_actuals_inv = scaler_target.inverse_transform(y.reshape(-1, 1))

# Plot full dataset predictions
fig, ax = plt.subplots(figsize=(16, 8))

dates_for_plot = df['Date'].iloc[SEQ_LENGTH:].reset_index(drop=True)
ax.plot(dates_for_plot, all_actuals_inv, label='Actual Price', color='#2E86AB', linewidth=1.5, alpha=0.8)
ax.plot(dates_for_plot, all_predictions_inv, label='Predicted Price', color='#EF476F', linewidth=1.5, alpha=0.8)

# Mark train/validation/test regions
train_end_date = dates_for_plot.iloc[train_size]
val_end_date = dates_for_plot.iloc[train_size + val_size]
ax.axvline(x=train_end_date, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Train/Val Split')
ax.axvline(x=val_end_date, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Val/Test Split')

ax.fill_betweenx([all_actuals_inv.min(), all_actuals_inv.max()], 
                  dates_for_plot.min(), train_end_date, alpha=0.1, color='green', label='Training Set')
ax.fill_betweenx([all_actuals_inv.min(), all_actuals_inv.max()], 
                  train_end_date, val_end_date, alpha=0.1, color='yellow', label='Validation Set')
ax.fill_betweenx([all_actuals_inv.min(), all_actuals_inv.max()], 
                  val_end_date, dates_for_plot.max(), alpha=0.1, color='red', label='Test Set')

ax.set_title('TSLA Stock Price Prediction - Full Dataset', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/06_full_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: output/06_full_prediction.png")

# Save prediction results to CSV
results_df = pd.DataFrame({
    'Date': df['Date'].iloc[SEQ_LENGTH:].values,
    'Actual_Price': all_actuals_inv.flatten(),
    'Predicted_Price': all_predictions_inv.flatten(),
    'Error': (all_predictions_inv - all_actuals_inv).flatten(),
    'Percentage_Error': ((all_predictions_inv - all_actuals_inv) / all_actuals_inv * 100).flatten()
})
results_df.to_csv('output/prediction_results.csv', index=False)
print("Saved prediction results: output/prediction_results.csv")

# ==============================================================================
# SECTION 7: FUTURE PRICE PREDICTION
# ==============================================================================
print("\n" + "=" * 60)
print("7. Future Price Prediction")
print("=" * 60)


def predict_future(model, last_sequence, scaler_features, scaler_target, days=10):
    """
    Predict future stock prices using rolling prediction approach.
    
    This function uses the trained model to predict future prices by
    iteratively feeding predictions back as inputs for subsequent predictions.
    
    Parameters:
    -----------
    model : nn.Module
        Trained prediction model
    last_sequence : numpy.ndarray
        Last sequence of scaled features from historical data
    scaler_features : MinMaxScaler
        Scaler used for features
    scaler_target : MinMaxScaler
        Scaler used for target variable
    days : int
        Number of days to predict into the future
        
    Returns:
    --------
    numpy.ndarray
        Predicted prices in original scale
    """
    model.eval()
    predictions = []
    current_seq = last_sequence.copy()
    
    with torch.no_grad():
        for _ in range(days):
            # Prepare input tensor
            x_input = torch.FloatTensor(current_seq).unsqueeze(0).to(device)
            
            # Make prediction
            pred = model(x_input)
            predictions.append(pred.cpu().numpy()[0, 0])
            
            # Update sequence for next prediction (rolling window)
            new_row = current_seq[-1].copy()
            # Update Close price feature (index 3) with prediction
            new_row[3] = pred.cpu().numpy()[0, 0]
            current_seq = np.vstack([current_seq[1:], new_row])
    
    # Inverse transform predictions to original scale
    predictions = np.array(predictions)
    predictions_inv = scaler_target.inverse_transform(predictions.reshape(-1, 1))
    return predictions_inv


# Predict next 10 days
last_seq = scaled_features[-SEQ_LENGTH:]
future_predictions = predict_future(model, last_seq, scaler_features, scaler_target, days=10)

print("\nFuture 10-Day Price Predictions:")
last_date = df['Date'].iloc[-1]
for i, pred in enumerate(future_predictions.flatten()):
    future_date = last_date + pd.Timedelta(days=i+1)
    print(f"  {future_date.strftime('%Y-%m-%d')}: ${pred:.2f}")

# Plot future predictions
fig, ax = plt.subplots(figsize=(14, 6))

# Plot recent 60 days of historical data
recent_data = df.tail(60)
ax.plot(recent_data['Date'], recent_data['Close'], label='Historical Price', 
        color='#2E86AB', linewidth=2, marker='o', markersize=3)

# Plot future predictions
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10, freq='D')
ax.plot(future_dates, future_predictions, label='Future Prediction', 
        color='#EF476F', linewidth=2, linestyle='--', marker='s', markersize=6)

# Add connecting line between last historical point and first prediction
ax.plot([last_date, future_dates[0]], [recent_data['Close'].iloc[-1], future_predictions[0][0]], 
        color='gray', linestyle=':', linewidth=2)

ax.axvline(x=last_date, color='green', linestyle='--', alpha=0.7, label='Prediction Start')
ax.set_title('TSLA Stock Price - Future 10-Day Prediction', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price (USD)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/07_future_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: output/07_future_prediction.png")

# Save future predictions to CSV
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Price': future_predictions.flatten()
})
future_df.to_csv('output/future_predictions.csv', index=False)
print("Saved future predictions: output/future_predictions.csv")

# ==============================================================================
# SECTION 8: SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("Project Summary")
print("=" * 60)

print(f"""
Project: TSLA Stock Price Prediction (LSTM + Transformer)
==================================================

Data Overview:
  - Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}
  - Sample Count: {len(df)}
  - Feature Count: {len(feature_columns)}

Model Configuration:
  - Sequence Length: {SEQ_LENGTH} days
  - LSTM Layers: 2 (Bidirectional)
  - Transformer Layers: 2
  - Hidden Dimension: 128
  - Attention Heads: 4

Training Configuration:
  - Train/Val/Test Split: 70%/15%/15%
  - Batch Size: {BATCH_SIZE}
  - Max Epochs: {EPOCHS}
  - Optimizer: AdamW

Test Set Performance:
  - RMSE: {rmse:.4f}
  - MAE: {mae:.4f}
  - R²: {r2:.4f}
  - MAPE: {mape:.2f}%

Output Files:
  - output/01_stock_overview.png - Stock Price Overview
  - output/02_data_analysis.png - Data Analysis
  - output/03_macd_analysis.png - MACD Analysis
  - output/04_training_curves.png - Training Curves
  - output/05_prediction_results.png - Prediction Results
  - output/06_full_prediction.png - Full Dataset Prediction
  - output/07_future_prediction.png - Future Prediction
  - output/best_model.pth - Model Weights
  - output/evaluation_metrics.csv - Evaluation Metrics
  - output/prediction_results.csv - Prediction Results Data
  - output/future_predictions.csv - Future Predictions Data
""")

print("=" * 60)
print("Project Completed Successfully!")
print("=" * 60)

