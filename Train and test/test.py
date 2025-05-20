import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             classification_report, confusion_matrix)
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.attention_weights = None

    def forward(self, values, keys, query, mask=None):
        N, value_len, _ = values.shape
        _, key_len, _ = keys.shape
        _, query_len, _ = query.shape
        
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        
        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)
        self.attention_weights = attention
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.embed_size)
        out = self.fc_out(out)
        return out

class EnhancedProteinModelCNNFirst(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=512, output_dim=1,
                 num_lstm_layers=3, num_attention_heads=8, num_filters=128,
                 kernel_size=5, num_fc_layers=2):
        super(EnhancedProteinModelCNNFirst, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters,
                                kernel_size=kernel_size, padding=kernel_size // 2)
        self.batch_norm_conv = nn.BatchNorm1d(num_filters)
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=hidden_dim,
                            num_layers=num_lstm_layers, batch_first=True,
                            dropout=0.5, bidirectional=True)
        self.attention = SelfAttention(embed_size=hidden_dim * 2, heads=num_attention_heads)
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
        # x shape: [batch, seq_len, embedding_dim]
        x = x.permute(0, 2, 1)  # to [batch, embed_dim, seq_len]
        x = self.conv1d(x)
        x = self.batch_norm_conv(x)
        x = x.permute(0, 2, 1)  # back to [batch, seq_len, filters]

        x, _ = self.lstm(x)
        x = self.attention(x, x, x)
        x = self.pool(x.permute(0, 2, 1)).squeeze(2)  # [batch, hidden_dim*2]

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

    # Save confusion matrix
    cm_df = pd.DataFrame(cm,
                         index=["Actual Non-Allergen", "Actual Allergen"],
                         columns=["Predicted Non-Allergen", "Predicted Allergen"])
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")

def evaluate(test_embeddings_path, test_labels_path, model_path, embedding_dim, batch_size=16):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_embeddings = np.load(test_embeddings_path, allow_pickle=True)
    test_labels = np.load(test_labels_path)

    test_data = torch.tensor(test_embeddings, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EnhancedProteinModelCNNFirst(embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()

    all_labels = []
    all_preds = []
    all_probs = []
    test_loss = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    test_loss /= len(test_loader)
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()
    all_probs = np.array(all_probs).flatten()

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    specificity = recall_score(all_labels, all_preds, pos_label=0)
    auc_roc = roc_auc_score(all_labels, all_probs)

    metrics = {
        "Test Loss": test_loss,
        "Accuracy": accuracy,
        "F1 Score (weighted)": f1,
        "Matthews Corr Coef": mcc,
        "Precision (weighted)": precision,
        "Recall (weighted)": recall,
        "Specificity": specificity,
        "AUC ROC": auc_roc
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Protein Allergenicity Model")
    parser.add_argument("--test_embeddings", type=str, required=True, help="Path to test embeddings numpy file (.npy)")
    parser.add_argument("--test_labels", type=str, required=True, help="Path to test labels numpy file (.npy)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model checkpoint (.pt)")
    parser.add_argument("--embedding_dim", type=int, required=True, help="Embedding dimension size (e.g., 1024 for SeqVec)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader")

    args = parser.parse_args()

    set_seed()

    evaluate(args.test_embeddings, args.test_labels, args.model_path, args.embedding_dim, args.batch_size)