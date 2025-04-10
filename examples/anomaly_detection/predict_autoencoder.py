import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

# 加载模型定义
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
        return self.decoder(self.encoder(x))

# 加载模型和 scaler（只加载一次，作为全局）
scaler = joblib.load("../../weaver/model/autoencoder_scaler_best.pkl")
model = Autoencoder(input_dim=3)
model.load_state_dict(torch.load("../../weaver/model/autoencoder_model_best.pt"))
model.eval()

# 🚀 核心函数
def predict_anomaly(df, threshold=None, return_score=True):
    # 1. 获取输入特征
    X = df[["heart_rate", "sleep_hours", "step_count"]].values
    X_scaled = scaler.transform(X)

    # 2. 计算重构误差
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        recon = model(X_tensor)
        mse = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()

    # 3. 设置阈值（默认取 95% 分位数）
    if threshold is None:
        threshold = np.percentile(mse, 95)

    # 4. 标记是否异常
    is_anomaly = (mse > threshold).astype(int)

    # 5. 返回新表（可选择是否包含 MSE 分数）
    result = df.copy()
    result["reconstruction_error"] = mse
    result["is_anomaly"] = is_anomaly

    if return_score:
        return result
    else:
        return result["is_anomaly"]
