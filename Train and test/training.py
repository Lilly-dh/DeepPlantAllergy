import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader,RandomSampler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score,recall_score, roc_auc_score, confusion_matrix
import numpy as np
import time
import psutil
import os
import pandas as pd
import random
import gc
from datetime import datetime


print(torch.__version__)  # Should output 1.8.1
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.get_device_name(0))  # Should return the name of your GPU
torch.cuda.empty_cache()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
train_roc_aucs, val_roc_aucs = [], []
cpu_mems, gpu_mem_trains, gpu_mem_vals, infer_times, epoch_times = [], [], [], [],[]
tps, tns, fps, fns = [], [], [], []

# Create timestamp string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args():
    parser = argparse.ArgumentParser(description="Train protein allergenicity model")
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data .npz file')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data .npz file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--embedding_dim', type=int, required=True, help='Embedding dimension size')
    parser.add_argument('--model_type', type=str, default="full",
                        choices=["full", "no_cnn", "no_lstm", "no_attention"],
                        help="Model architecture type to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

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
            
        attention = F.softmax(energy / (self.embed_size ** 0.5), dim=3)
        self.attention_weights = attention
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)
        
        return self.fc_out(out)
    
    def get_attention_weights(self):
        return self.attention_weights


# 1. Full model: CNN + LSTM + Attention
class EnhancedProteinModelFull(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_lstm_layers=2, num_fc_layers=2, num_attention_heads=16,
                 num_filters=64, kernel_size=3):
        super().__init__()
        
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                                kernel_size=kernel_size, padding=kernel_size // 2)
        self.batch_norm_conv = nn.BatchNorm1d(num_filters)
        
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_dim,
                            num_layers=num_lstm_layers, batch_first=True,
                            dropout=0.5, bidirectional=True)
        
        self.attention = SelfAttention(embed_size=hidden_dim * 2,
                                       heads=num_attention_heads)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_layers = nn.ModuleList()
        self.batch_norm_fc = nn.ModuleList()
        input_dim = hidden_dim * 2
        
        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norm_fc.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.batch_norm_conv(x)
        x = x.permute(0, 2, 1)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Attention
        x = self.attention(x, x, x)
        
        # Pooling
        x = self.pool(x.permute(0, 2, 1)).squeeze()
        
        # Fully connected
        for fc, bn in zip(self.fc_layers, self.batch_norm_fc):
            x = self.relu(bn(fc(x)))
            x = self.dropout(x)
        
        x = self.fc_output(x)
        return x


# 2. No CNN: LSTM + Attention
class EnhancedProteinModelNoCNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_lstm_layers=2, num_fc_layers=2, num_attention_heads=16):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_lstm_layers, batch_first=True,
                            dropout=0.5, bidirectional=True)
        
        self.attention = SelfAttention(embed_size=hidden_dim * 2,
                                       heads=num_attention_heads)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_layers = nn.ModuleList()
        self.batch_norm_fc = nn.ModuleList()
        input_dim = hidden_dim * 2
        
        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norm_fc.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # LSTM
        x, _ = self.lstm(x)
        
        # Attention
        x = self.attention(x, x, x)
        
        # Pooling
        x = self.pool(x.permute(0, 2, 1)).squeeze()
        
        # Fully connected
        for fc, bn in zip(self.fc_layers, self.batch_norm_fc):
            x = self.relu(bn(fc(x)))
            x = self.dropout(x)
        
        x = self.fc_output(x)
        return x


# 3. No LSTM: CNN + Attention
class EnhancedProteinModelNoLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_fc_layers=2, num_attention_heads=16,
                 num_filters=64, kernel_size=3):
        super().__init__()
        
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                                kernel_size=kernel_size, padding=kernel_size // 2)
        self.batch_norm_conv = nn.BatchNorm1d(num_filters)
        
        self.attention = SelfAttention(embed_size=num_filters,
                                       heads=num_attention_heads)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_layers = nn.ModuleList()
        self.batch_norm_fc = nn.ModuleList()
        input_dim = num_filters
        
        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norm_fc.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.batch_norm_conv(x)
        x = x.permute(0, 2, 1)
        
        # Attention
        x = self.attention(x, x, x)
        
        # Pooling
        x = self.pool(x.permute(0, 2, 1)).squeeze()
        
        # Fully connected
        for fc, bn in zip(self.fc_layers, self.batch_norm_fc):
            x = self.relu(bn(fc(x)))
            x = self.dropout(x)
        
        x = self.fc_output(x)
        return x


# 4. No Attention: CNN + LSTM
class EnhancedProteinModelNoAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 num_lstm_layers=2, num_fc_layers=2,
                 num_filters=64, kernel_size=3):
        super().__init__()
        
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                                kernel_size=kernel_size, padding=kernel_size // 2)
        self.batch_norm_conv = nn.BatchNorm1d(num_filters)
        
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_dim,
                            num_layers=num_lstm_layers, batch_first=True,
                            dropout=0.5, bidirectional=True)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_layers = nn.ModuleList()
        self.batch_norm_fc = nn.ModuleList()
        input_dim = hidden_dim * 2
        
        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_dim, hidden_dim))
            self.batch_norm_fc.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.batch_norm_conv(x)
        x = x.permute(0, 2, 1)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Pooling
        x = self.pool(x.permute(0, 2, 1)).squeeze()
        
        # Fully connected
        for fc, bn in zip(self.fc_layers, self.batch_norm_fc):
            x = self.relu(bn(fc(x)))
            x = self.dropout(x)
        
        x = self.fc_output(x)
        return x

def train_model_with_metrics(model, train_loader, test_loader, optimizer, criterion, scheduler, num_epochs, seed):
    global best_val_loss, patience_counter, best_epoch, best_model_weights
    model.to(device)
    model.train()
    process = psutil.Process(os.getpid())  # CPU memory tracking

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        start_cpu = time.process_time()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        running_train_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        all_train_probs = []

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
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

        # Peak GPU mem for training
        gpu_mem_train_mb = None
        if device.type == "cuda" and torch.cuda.is_available():
            gpu_mem_train_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        train_mcc = matthews_corrcoef(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted')
        train_roc_auc = roc_auc_score(all_train_labels, all_train_probs)

        model.eval()
        running_val_loss = 0.0
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()  # reset before validation to measure separately

        infer_start_time = time.time()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds)
                all_val_probs.extend(probs)
        infer_time = time.time() - infer_start_time

        # Peak GPU mem for validation
        gpu_mem_val_mb = None
        if device.type == "cuda" and torch.cuda.is_available():
            gpu_mem_val_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        val_loss = running_val_loss / len(test_loader)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        val_mcc = matthews_corrcoef(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
        val_roc_auc = roc_auc_score(all_val_labels, all_val_probs)

        tn, fp, fn, tp = confusion_matrix(all_val_labels, all_val_preds).ravel()
        cpu_mem_mb = process.memory_info().rss / (1024 ** 2)  # in MB
        epoch_time = time.time() - epoch_start_time
        end_cpu = time.process_time()  # CPU time end
        cpu_time = end_cpu - start_cpu  # CPU time elapsed


        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Train ROC-AUC: {train_roc_auc:.4f}, Val ROC-AUC: {val_roc_auc:.4f}, "
              f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}, "
              f"CPU Mem: {cpu_mem_mb:.1f} MB, "
              f"GPU Train Mem: {gpu_mem_train_mb if gpu_mem_train_mb else 0:.1f} MB, "
              f"GPU Val Mem: {gpu_mem_val_mb if gpu_mem_val_mb else 0:.1f} MB, "
              f"Infer Time: {infer_time:.2f}s, Epoch Time: {epoch_time:.2f}s")

        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_f1s.append(train_f1)
        train_mccs.append(train_mcc)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_roc_aucs.append(train_roc_auc)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1s.append(val_f1)
        val_mccs.append(val_mcc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_roc_aucs.append(val_roc_auc)

        cpu_mems.append(cpu_mem_mb)
        gpu_mem_trains.append(gpu_mem_train_mb if gpu_mem_train_mb else 0)
        gpu_mem_vals.append(gpu_mem_val_mb if gpu_mem_val_mb else 0)
        infer_times.append(infer_time)
        epoch_times.append(epoch_time)
        cpu_times.append(cpu_time)
        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)

        # Step the scheduler if provided
        if scheduler:
            scheduler.step(val_loss)  # Assuming ReduceLROnPlateau
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Current Learning Rate: {current_lr}")
        learning_rates.append(current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_weights = model.state_dict()
            best_model_path = f"best_model_epoch{epoch+1}_{timestamp}_seed_{seed}.pt"
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

def load_npz_data(path):
    data = np.load(path)
    return data['embeddings'], data['labels'], data['sequence_ids']

def main():
    args = parse_args()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    

    # Load embeddings and labels
    train_embs, train_labels, _ = load_npz_data(args.train_data)
    X_train = torch.tensor(train_embs, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size, sampler=RandomSampler(train_dataset, generator=torch.Generator().manual_seed(args.seed)))
    del train_embs, train_labels, X_train
    gc.collect()
    
    test_embs, test_labels, _ = load_npz_data(args.val_data)
    X_test = torch.tensor(test_embs, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)
    test_dataset = TensorDataset(X_test, y_test)    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
    del test_embs, test_labels, X_test
    gc.collect()

    # Initialize model, criterion, optimizer, scheduler
    # Create model instance
    
    if args.model_type == "full":
        model = EnhancedProteinModelFull(
            embedding_dim=args.embedding_dim, hidden_dim=128, output_dim=1, num_lstm_layers=3, num_filters=64, num_attention_heads=8, num_fc_layers=3, kernel_size=5)
    elif args.model_type == "no_cnn":
        model = EnhancedProteinModelNoCNN(embedding_dim=args.embedding_dim, hidden_dim=128, output_dim=1,num_lstm_layers=3, num_fc_layers=3, num_attention_heads=8)
    elif args.model_type == "no_lstm":
        model = EnhancedProteinModelNoLSTM(embedding_dim=args.embedding_dim, hidden_dim=128, output_dim=1,num_fc_layers=3, num_attention_heads=8, num_filters=64, kernel_size=5)
    elif args.model_type == "no_attention":
        model = EnhancedProteinModelNoAttention(embedding_dim=args.embedding_dim, hidden_dim=128, output_dim=1,num_lstm_layers=3, num_filters=64,num_fc_layers=3, kernel_size=5)
    else:
        raise ValueError(f"Unknown model_type {args.model_type}")

    
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=5)

    # Train model
    train_model_with_metrics(model, train_loader, test_loader, optimizer, criterion, scheduler, args.num_epochs, args.seed)
    
    # Create DataFrames from metrics lists
    train_metrics = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies,
        'Train F1 Score': train_f1s,
        'Train MCC': train_mccs,
        'Train Precision': train_precisions,
        'Train Recall':train_recalls,
        'Train ROC-AUC': train_roc_aucs,
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
        'Validation ROC-AUC': val_roc_aucs,
        'TP': tps,
        'TN': tns,
        'FP': fps,
        'FN': fns,
        'CPU Memory (MB)': cpu_mems,
        'GPU Memory Train (MB)': gpu_mem_trains,
        'GPU Memory Val (MB)': gpu_mem_vals,
        'Inference Time (s)': infer_times,
        'Epoch Time (s)': epoch_times,
        'CPU Time':cpu_times
    })


    # Concatenate DataFrames column-wise (axis=1)
    combined_metrics = pd.concat([train_metrics, val_metrics.drop(columns=['Epoch'])], axis=1)

    # Save the combined DataFrame to a single CSV file
    combined_metrics.to_csv(f"training_metrics_{args.num_epochs}_{timestamp}_{args.model_type}_seed_{args.seed}.csv", index=False)
    print("Saved training metrics to training_metrics.csv")


if __name__ == "__main__":
    main()
