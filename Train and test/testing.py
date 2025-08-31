import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import time
import psutil
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             classification_report, confusion_matrix)
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def save_metrics_and_cm(metrics_dict, cm, output_dir, model_name):
    # Save metrics
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

def load_npz_data(path):
    data = np.load(path)
    return data['embeddings'], data['labels'], data['sequence_ids']

class ProteinDataset(Dataset):
    def __init__(self, embeddings, labels, seq_ids):
        self.embeddings = embeddings
        self.labels = labels
        self.seq_ids = seq_ids
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        label = self.labels[idx]
        seq_id = self.seq_ids[idx]
        return emb, label, seq_id

def evaluate(test_data_path, model_path, embedding_dim, batch_size=16, model_type=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_embs, test_labels, test_seq_ids = load_npz_data(test_data_path)

    # Decode bytes to str if needed
    test_seq_ids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in test_seq_ids]

    test_data = torch.tensor(test_embs, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

    test_dataset = ProteinDataset(test_data, test_labels, test_seq_ids)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if model_type == "full":
        model = EnhancedProteinModelFull(
            embedding_dim=embedding_dim, hidden_dim=128, output_dim=1, 
            num_lstm_layers=3, num_filters=64, num_attention_heads=8, num_fc_layers=3, kernel_size=5)
    elif model_type == "no_cnn":
        model = EnhancedProteinModelNoCNN(
            embedding_dim=embedding_dim, hidden_dim=128, output_dim=1,
            num_lstm_layers=3, num_fc_layers=3, num_attention_heads=8)
    elif model_type == "no_lstm":
        model = EnhancedProteinModelNoLSTM(
            embedding_dim=embedding_dim, hidden_dim=128, output_dim=1,
            num_fc_layers=3, num_attention_heads=8, num_filters=64, kernel_size=5)
    elif model_type == "no_attention":
        model = EnhancedProteinModelNoAttention(
            embedding_dim=embedding_dim, hidden_dim=128, output_dim=1,
            num_lstm_layers=3, num_filters=64, num_fc_layers=3, kernel_size=5)
    else:
        raise ValueError(f"Unknown model_type {model_type}")    

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    all_labels = []
    all_preds = []
    all_probs = []
    all_seq_ids = []
    test_loss = 0
    
    process = psutil.Process()
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    infer_start_time = time.time()

    with torch.no_grad():
        for inputs, labels, seq_ids in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_seq_ids.extend(seq_ids)

    infer_time = time.time() - infer_start_time
    cpu_mem_mb = process.memory_info().rss / (1024 ** 2)
    
    if device.type == "cuda":
        gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        gpu_mem_mb = 0

    test_loss /= len(test_loader)
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()
    all_seq_ids = [str(s) for s in all_seq_ids]

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    specificity = recall_score(all_labels, all_preds, pos_label=0)
    auc_roc = roc_auc_score(all_labels, all_probs)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    metrics = {
        "Test Loss": test_loss,
        "Accuracy": accuracy,
        "F1 Score (weighted)": f1,
        "Matthews Corr Coef": mcc,
        "Precision (weighted)": precision,
        "Recall (weighted)": recall,
        "Specificity": specificity,
        "AUC ROC": auc_roc,
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "CPU Memory (MB)": cpu_mem_mb,
        "GPU Memory (MB)": gpu_mem_mb,
        "Inference Time (s)": infer_time
    }

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Matthews Corr Coef: {mcc:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"AUC ROC: {auc_roc:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Allergen', 'Allergen']))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Save outputs
    output_dir = os.path.dirname(model_path)
    if output_dir == "":
        output_dir = "."
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    save_metrics_and_cm(metrics, cm, output_dir, model_name)

    # Save per-sequence predictions
    results_df = pd.DataFrame({
        "Sequence_ID": all_seq_ids,
        "True_Label": all_labels,
        "Predicted_Probability": all_probs,
        "Predicted_Class": all_preds
    })
    results_csv_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Predictions saved to: {results_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Protein Allergenicity Model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data(.npz)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model checkpoint (.pt)")
    parser.add_argument("--embedding_dim", type=int, required=True, help="Embedding dimension size (e.g., 1024 for SeqVec)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader")
    parser.add_argument("--model_type", type=str, default="full",
                        choices=["full", "no_cnn", "no_lstm", "no_attention"],
                        help="Model variant to load")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    set_seed(args.seed)

    evaluate(args.test_data, args.model_path, args.embedding_dim, args.batch_size, args.model_type)

