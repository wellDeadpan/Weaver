import copy
from model.config import FEATURES, OUTCOMES
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Step 1:
df = pd.read_json("synthetic_health_anomaly_data.json", orient="records")
print(df.head())

# Step 2:
X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y1 = df[OUTCOMES[0]]
y2 = df[OUTCOMES[1]]

# Step 2: 划分 Train / Val / Test
X_trainval, X_test, y1_trainval,  y1_test = train_test_split(X_scaled, y1, stratify=y1, test_size=0.2, random_state=42)
X_train, X_val, y1_train, y1_val = train_test_split(X_trainval, y1_trainval, stratify=y1_trainval, test_size=0.2, random_state=42)

# Step 3: 标准化（只fit训练集）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Step 4: 转为 PyTorch 张量 + Dataloader
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor), batch_size=32, shuffle=False)

# Step 5: 定义Autoencoder模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

# Step 6: 初始化模型与优化器
model = Autoencoder(input_dim=X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 7: 加入 Early Stopping 机制
n_epochs = 100
best_loss = float('inf')
best_model = None
patience = 0
max_patience = 10
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    train_epoch_loss = 0
    for batch in train_loader:
        x_batch = batch[0]
        optimizer.zero_grad()
        recon = model(x_batch)
        loss = criterion(recon, x_batch)
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.item()

    model.eval()
    val_epoch_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x_batch = batch[0]
            recon = model(x_batch)
            val_loss = criterion(recon, x_batch)
            val_epoch_loss += val_loss.item()

    train_losses.append(train_epoch_loss)
    val_losses.append(val_epoch_loss)

    print(f"Epoch {epoch + 1}: Train Loss = {train_epoch_loss:.4f}, Val Loss = {val_epoch_loss:.4f}")

    if val_epoch_loss < best_loss:
        best_loss = val_epoch_loss
        best_model = copy.deepcopy(model.state_dict())
        patience = 0
    else:
        patience += 1
        if patience > max_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

# Step 8: 恢复最佳模型 & 保存
model.load_state_dict(best_model)
torch.save(model.state_dict(), "autoencoder_model_best.pt")
joblib.dump(scaler, "autoencoder_scaler_best.pkl")

# Step 9: 可视化训练曲线
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss Curve with Early Stopping")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()


# 预测并计算重构误差
X_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    X_recon = model(X_tensor)
    mse = torch.mean((X_tensor - X_recon)**2, dim=1).numpy()

# 绘制 ROC 和 PR 曲线
fpr, tpr, _ = roc_curve(y1_test, mse)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y1_test, mse)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")

plt.tight_layout()
plt.show()
