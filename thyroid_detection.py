import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. Configuration & Data Preparation
# ============================================================================
class Config:
    TRAIN_PATH = 'thyroid/thyroid/train-set.csv'
    TEST_PATH = 'thyroid/thyroid/test-set.csv'
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-3
    SEED = 42
    INPUT_DIM = 6
    HIDDEN_DIMS = [32, 16, 8]  # Encoder structure, Decoder will be symmetric

def load_and_preprocess_data():
    """Load data, standardize, and convert to PyTorch tensors."""
    # Load raw data
    train_df = pd.read_csv(Config.TRAIN_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)
    
    print("Data Statistics:")
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Test set disease prevalence: {test_df['label'].mean():.2%}")
    
    # Extract features and labels
    # Train set only has features (all normal)
    X_train = train_df.values
    
    # Test set has features + label
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    # Standardization
    # Fit scaler ONLY on training data (normal samples) to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to Tensors
    train_tensor = torch.FloatTensor(X_train_scaled)
    test_tensor = torch.FloatTensor(X_test_scaled)
    test_labels = torch.FloatTensor(y_test)
    
    return train_tensor, test_tensor, test_labels

# ============================================================================
# 2. Model Definition: Autoencoder
# ============================================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (Symmetric)
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        for i, h_dim in enumerate(hidden_dims_reversed):
            out_dim = hidden_dims_reversed[i+1] if i < len(hidden_dims_reversed)-1 else input_dim
            decoder_layers.append(nn.Linear(h_dim, out_dim))
            if i < len(hidden_dims_reversed)-1:
                decoder_layers.append(nn.ReLU())
            # No activation on final layer or Tanh/Sigmoid depending on data range (StandardScaler used, so linear is fine)
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ============================================================================
# 3. Training & Evaluation
# ============================================================================
def train_model(model, train_loader, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    train_losses = []
    
    model.train()
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x) # Reconstruction Loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{Config.EPOCHS}], Loss: {avg_loss:.6f}")
            
    return train_losses

def evaluate_model(model, train_tensor, test_tensor, test_labels, device):
    model.eval()
    criterion = nn.MSELoss(reduction='none') # Calculate loss per element
    
    with torch.no_grad():
        # Get reconstruction error for Training set (to determine threshold)
        train_input = train_tensor.to(device)
        train_output = model(train_input)
        # Mean over features for each sample
        train_loss = criterion(train_output, train_input).mean(dim=1).cpu().numpy()
        
        # Get reconstruction error for Test set
        test_input = test_tensor.to(device)
        test_output = model(test_input)
        test_loss = criterion(test_output, test_input).mean(dim=1).cpu().numpy()
        
    return train_loss, test_loss

def analyze_results(train_errors, test_errors, test_labels):
    """
    Analyze performance and plot results.
    """
    # 1. Determine Threshold
    # Strategy: using 95th or 99th percentile of training errors (allowing some noise in train data)
    # Alternatively: Mean + 3 Std
    threshold_percentile = np.percentile(train_errors, 95)
    threshold_std = train_errors.mean() + 2 * train_errors.std()
    
    threshold = threshold_std # Let's use 2-sigma rule for now
    
    print("\n--- Anomaly Detection Analysis ---")
    print(f"Threshold (Mean + 2*Std of Normal): {threshold:.6f}")
    
    # 2. Key Metrics
    # Calculate ROC-AUC
    roc_auc = roc_auc_score(test_labels, test_errors)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Predictions based on threshold
    y_pred = (test_errors > threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, y_pred, average='binary')
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    cm = confusion_matrix(test_labels, y_pred)
    df_cm = pd.DataFrame(cm, index=['Normal (0)', 'Disease (1)'], columns=['Pred Normal', 'Pred Disease'])
    print("\nConfusion Matrix:")
    print(df_cm)

    # 3. Visualization
    plt.figure(figsize=(12, 5))
    
    # Histograms
    plt.subplot(1, 2, 1)
    sns.histplot(train_errors, color='blue', label='Train (Normal)', kde=True, stat="density", alpha=0.3)
    sns.histplot(test_errors[test_labels==0], color='green', label='Test (Normal)', kde=True, stat="density", alpha=0.3)
    sns.histplot(test_errors[test_labels==1], color='red', label='Test (Disease)', kde=True, stat="density", alpha=0.3)
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.legend()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, test_errors)
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('thyroid_analysis.png')
    print("\nAnalysis plot saved as 'thyroid_analysis.png'")

# ============================================================================
# Main Execution
# ============================================================================
def main():
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    train_tensor, test_tensor, test_labels = load_and_preprocess_data()
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # 2. Build Model
    model = Autoencoder(Config.INPUT_DIM, Config.HIDDEN_DIMS).to(device)
    # print(model)
    
    # 3. Train
    print("\nTraining Autoencoder on Normal Data...")
    train_model(model, train_loader, device)
    
    # 4. Evaluate
    print("\nEvaluating on Test Set...")
    train_errors, test_errors = evaluate_model(model, train_tensor, test_tensor, test_labels, device)
    
    # 5. Analyze
    analyze_results(train_errors, test_errors, test_labels.numpy())

if __name__ == '__main__':
    main()
