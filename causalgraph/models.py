
from __future__ import annotations
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from tqdm import tqdm
from .utils import compute_c_from_dag
from .CausalLogisticRegression import CausalLogisticRegression, CausalLinearRegression

# Fixed hyperparameters
EPOCHS = 40000           # rely on early stopping
BATCH_SIZE = 256
LR_MLP = 4e-5
WEIGHT_DECAY = 0.0
MAX_PATIENCE = 10

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TorchTabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

# MLP with hard-coded architecture
class MLP(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def _compute_metrics(task: str, y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, logits.ravel())))
        return {"rmse": rmse}
    else:
        if logits.ndim == 1 or logits.shape[1] == 1:
            z = np.clip(logits.ravel(), -80, 80)  # 防止 exp 溢位/下溢
            prob = 1.0 / (1.0 + np.exp(-z))
            try:
                auc = float(roc_auc_score(y_true, prob))
            except Exception:
                auc = float("nan")
        else:
            exps = np.exp(logits - logits.max(axis=1, keepdims=True))
            prob = exps / exps.sum(axis=1, keepdims=True)
            try:
                auc = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
            except Exception:
                print("exception")
                auc = float("nan")
        return {"auroc": auc}

def _train_eval_sklearn(
    X_train,
    X_valid,
    y_train_raw,
    y_valid_raw,
    task,
    c=None,
    lambda_causal=1.0,
):
    # ------------------------
    # ① regression（含 causal）
    # ------------------------
    if task == "regression":
        y_train_f = y_train_raw.astype(float)
        y_valid_f = y_valid_raw.astype(float)

        if c is None:  # normal linear regression
            model = LinearRegression()
            model.fit(X_train, y_train_f)
            preds = model.predict(X_valid)
            return _compute_metrics(task, y_valid_f, preds)

        # causal regression
        model = CausalLinearRegression(
            c=c,
            lam=lambda_causal,
            lr=1e-3,
            max_iter=2000,
        )
        model.fit(X_train, y_train_f)
        preds = model.predict(X_valid)
        return _compute_metrics(task, y_valid_f, preds)

    # ------------------------
    # ② classification（保持你原本邏輯）
    # ------------------------

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw.astype(str))
    y_valid = le.transform(y_valid_raw.astype(str))

    if c is None:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_valid)
        return _compute_metrics(task, y_valid, proba)

    clf = CausalLogisticRegression(
        c=c,
        lam=lambda_causal,
        lr=1e-3,
        max_iter=2000,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_valid)
    return _compute_metrics(task, y_valid, proba)

def _train_eval_mlp(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train_raw: np.ndarray,
    y_valid_raw: np.ndarray,
    task: str,
) -> Dict[str, float]:
    device = _device()

    if task == "classification":
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw.astype(str))
        y_valid = le.transform(y_valid_raw.astype(str))
        num_classes = int(len(le.classes_))
    else:
        y_train = y_train_raw.astype(np.float32)
        y_valid = y_valid_raw.astype(np.float32)
        num_classes = 1

    ds_tr = TorchTabularDataset(X_train, y_train)
    ds_va = TorchTabularDataset(X_valid, y_valid)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    out_dim = 1 if task == "regression" else (1 if num_classes == 2 else num_classes)
    model = MLP(input_dim=X_train.shape[1], out_dim=out_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR_MLP, weight_decay=WEIGHT_DECAY)

    criterion = nn.MSELoss() if task == "regression" else (nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss())

    best_metric = {'rmse': 1e10, 'auroc': 0.0}
    patience = MAX_PATIENCE
    latest_metrics: Dict[str, float] = {}

    for epoch in tqdm(range(1, EPOCHS + 1), desc=f"Training (mlp, {task})", unit="epoch"):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            if task == "regression":
                yb = yb.float().to(device).view(-1, 1)
            else:
                yb = (yb.float().to(device).view(-1, 1) if isinstance(criterion, nn.BCEWithLogitsLoss)
                      else yb.long().to(device))
            optim.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            all_logits, all_y = [], []
            for xb, yb in dl_va:
                xb = xb.to(device)
                out = model(xb)
                logits = out.cpu().numpy() if (task == "classification" and isinstance(criterion, nn.CrossEntropyLoss)) else out.squeeze(-1).cpu().numpy()
                all_logits.append(logits)
                all_y.append(yb.numpy())
            logits_np = np.concatenate(all_logits, axis=0)
            y_np = np.concatenate(all_y, axis=0)
            latest_metrics = _compute_metrics(task, y_np, logits_np)

        if task == "regression":
            tqdm.write(f"Epoch {epoch:03d} | train_loss={epoch_loss/len(dl_tr):.4f} | val_RMSE={latest_metrics['rmse']:.4f}")
            if latest_metrics['rmse'] < best_metric['rmse']:
                best_metric['rmse'] = latest_metrics['rmse']; patience = MAX_PATIENCE
            else:
                patience -= 1; tqdm.write(f"No improvement. patience={patience}")
        else:
            tqdm.write(f"Epoch {epoch:03d} | train_loss={epoch_loss/len(dl_tr):.4f} | val_AUROC={latest_metrics['auroc']:.4f}")
            if latest_metrics['auroc'] > best_metric['auroc']:
                best_metric['auroc'] = latest_metrics['auroc']; patience = MAX_PATIENCE
            else:
                patience -= 1; tqdm.write(f"No improvement. patience={patience}")

        if patience == 0:
            break

    return {'rmse': best_metric['rmse']} if task == "regression" else {'auroc': best_metric['auroc']}

def train_and_eval(
    train_df,
    valid_df,
    X_train,
    X_valid,
    target,
    task="regression",
    model_type="linear",
    graph=None,
):
    y_train = train_df[target].values
    y_valid = valid_df[target].values

    if model_type == "linear":
        return _train_eval_sklearn(X_train, X_valid, y_train, y_valid, task)

    elif model_type == "mlp":
        return _train_eval_mlp(X_train, X_valid, y_train, y_valid, task)

    elif model_type == "causal_linear":
        if graph is None:
            raise ValueError("graph must be provided for causal_linear")

        # feature 順序 = X 的欄位順序（假設一致）
        variables = list(train_df.drop(columns=[target]).columns)

        c = compute_c_from_dag(
            graph=graph,
            variables=variables,
            target=target,
        )

        return _train_eval_sklearn(
            X_train,
            X_valid,
            y_train,
            y_valid,
            task,
            c=c,
            lambda_causal=1.0,
        )

    else:
        raise ValueError("model_type must be 'linear', 'mlp', or 'causal_linear'")
