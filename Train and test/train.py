import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader,RandomSampler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score,recall_score
import numpy as np
import time
import psutil
import os
import pandas as pd
import random
import gc

print(torch.__version__)  # Should output 1.8.1
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.get_device_name(0))  # Should return the name of your GPU
torch.cuda.empty_cache()

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)          # If using CUDA
torch.cuda.manual_seed_all(seed)      # If using multi-GPU setups
np.random.seed(seed)
random.seed(seed)

# Ensure that CUDA operations are deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set fixed parameters internally
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Metrics storage (global)
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
train_f1s, val_f1s = [], []
train_mccs, val_mccs = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []
learning_rates = []
cpu_times = []
best_val_loss = float('inf')
patience_counter = 0
best_epoch = 0
best_model_weights = None

def parse_args():
    parser = argparse.ArgumentParser(description="Train protein allergenicity model")
    parser.add_argument('--train_embs', type=str, required=True, help='Path to training embeddings .npy file')
    parser.add_argument('--train_labels', type=str, required=True, help='Path to training labels .npy file')
    parser.add_argument('--test_embs', type=str, required=True, help='Path to test embeddings .npy file')
    parser.add_argument('--test_labels', type=str, required=True, help='Path to test labels .npy file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--embedding_dim', type=int, required=True, help='Embedding dimension size')
    return parser.parse_args()

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embedding size must be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
        self.attention_weights = None
    
    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        self.attention_weights = attention
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)
        
        return self.fc_out(out)
    
    def get_attention_weights(self):
        return self.attention_weights

class EnhancedProteinModelCNNFirst(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_lstm_layers=2, num_fc_layers=2, num_attention_heads=16, num_filters=64, kernel_size=3):
        super(EnhancedProteinModelCNNFirst, self).__init__()
        
        # CNN
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.batch_norm_conv = nn.BatchNorm1d(num_filters)
        # LSTM
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_dim, num_layers=num_lstm_layers, batch_first=True, dropout=0.5, bidirectional=True)
        # Self-Attention
        self.attention = SelfAttention(embed_size=hidden_dim * 2, heads=num_attention_heads)
        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layers
        self.fc_layers = nn.ModuleList()
        self.batch_norm_fc = nn.ModuleList()
        input_dim = hidden_dim * 2

        for i in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norm_fc.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim

        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_length]
        x = self.conv1d(x)  # [batch_size, num_filters, seq_length]
        x = self.batch_norm_conv(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, num_filters]

        # LSTM
        x, _ = self.lstm(x)  # [batch_size, seq_length, hidden_dim*2]

        # Attention
        x = self.attention(x, x, x)  # [batch_size, seq_length, hidden_dim*2]

        # Pooling
        x = self.pool(x.permute(0, 2, 1)).squeeze()  # [batch_size, hidden_dim*2]

        # Fully Connected Layers
        for fc, bn in zip(self.fc_layers, self.batch_norm_fc):
            x = self.relu(bn(fc(x)))  # [batch_size, hidden_dim]
            x = self.dropout(x)

        # Output Layer
        x = self.fc_output(x)  # [batch_size, output_dim]
        return x


def train_model_with_metrics(model, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs):
    global best_val_loss, patience_counter, best_epoch, best_model_weights
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        cpu_start_time = time.process_time()
        process = psutil.Process(os.getpid())

        running_train_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        all_train_probs = []

        for inputs, labels in train_loader:
            inputs=inputs.to(device) 
            labels=labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds)
            all_train_probs.extend(probs)

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        train_mcc = matthews_corrcoef(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted')

        model.eval()
        running_val_loss = 0.0
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds)
                all_val_probs.extend(probs)

        val_loss = running_val_loss / len(test_loader)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        val_mcc = matthews_corrcoef(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
        cpu_time = time.process_time() - cpu_start_time


        print(f"⏳Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}, Train MCC: {train_mcc:.4f}, "
              f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val MCC: {val_mcc:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, "
              f"CPU Time: {cpu_time:.2f}s")

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_f1s.append(train_f1)
        train_mccs.append(train_mcc)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1s.append(val_f1)
        val_mccs.append(val_mcc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        cpu_times.append(cpu_time)

        # Step the scheduler if provided
        if scheduler:
            scheduler.step(val_loss)  # Assuming ReduceLROnPlateau
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']  # For non-scheduler cases

        # Log the learning rate
        print(f"Current Learning Rate: {current_lr}")
        learning_rates.append(current_lr)  # Append learning rate for the epoch

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_weights = model.state_dict()
            best_model_path = f"best_model_epoch{epoch+1}_training.pt"
            torch.save(best_model_weights, best_model_path)
            print(f"✅ Best model saved at epoch {epoch + 1}: {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"🚨 Early stopping patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("⏹️ Early stopping triggered. Stopping training.")
            break

        model.train()


def main():
    args = parse_args()

    # Load embeddings and labels
    train_embs = np.load(args.train_embs)
    train_labels = np.load(args.train_labels)
    X_train = torch.tensor(train_embs, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size, sampler=RandomSampler(train_dataset, generator=torch.Generator().manual_seed(42)))
    del train_embs, train_labels, X_train
    gc.collect()
    
    test_embs = np.load(args.test_embs)
    test_labels = np.load(args.test_labels)
    X_test = torch.tensor(test_embs, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)
    test_dataset = TensorDataset(X_test, y_test)    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
    del test_embs, test_labels, X_test
    gc.collect()

    # Initialize model, criterion, optimizer, scheduler
    # Create model instance
    model = EnhancedProteinModelCNNFirst(args.embedding_dim, hidden_dim=512, output_dim=1, num_lstm_layers=3, num_filters=128, num_attention_heads=8,num_fc_layers=2,kernel_size=5).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=5)

    # Train model
    train_model_with_metrics(model, train_loader, test_loader, optimizer, criterion, scheduler, args.num_epochs)

    # Create DataFrames from metrics lists
    train_metrics = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies,
        'Train F1 Score': train_f1s,
        'Train MCC': train_mccs,
        'Train Precision': train_precisions,
        'Train Recall':train_recalls,
        'Learning Rate': learning_rates
    })

    val_metrics = pd.DataFrame({
        'Epoch': range(1, len(val_losses) + 1),
        'Validation Loss': val_losses,
        'Validation Accuracy': val_accuracies,
        'Validation F1 Score': val_f1s,
        'Validation MCC': val_mccs,
        'Validation Precision': val_precisions,
        'Validation Recall':val_recalls,
        'CPU Time':cpu_times
    })


    # Concatenate DataFrames column-wise (axis=1)
    combined_metrics = pd.concat([train_metrics, val_metrics.drop(columns=['Epoch'])], axis=1)

    # Save the combined DataFrame to a single CSV file
    combined_metrics.to_csv(f'training_metrics_{args.num_epochs}.csv', index=False)
    print("📊 Saved training metrics to training_metrics.csv")


if __name__ == "__main__":
    main()
