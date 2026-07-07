import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path

PROCESSED_DIR = Path("./data/processed")
DATA_PATH = PROCESSED_DIR / "soc_time_series_features.parquet"
device = torch.device("cpu")


def prepare_soc_physics_features(window_size=30):
    print("Engineering physics-informed features and integrating Coulomb approximations...")
    df = pd.read_parquet(DATA_PATH)
    
    df['dt'] = df.groupby(['battery_id', 'cycle'])['time'].diff().fillna(0)
    df['amphour_spent'] = (df['current'] * df['dt']) / 3600.0
    df['cum_amphour_spent'] = df.groupby(['battery_id', 'cycle'])['amphour_spent'].cumsum()
    
    features = ['voltage', 'current', 'temperature', 'cum_amphour_spent']
    target = 'target_soc'
    
    train_df = df[df['battery_id'].isin(['B0005', 'B0006'])].copy()
    test_df = df[df['battery_id'] == 'B0007'].copy()
    
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])
    
    def create_windows(data, size):
        X, y = [], []
        for _, group in data.groupby(['battery_id', 'cycle']):
            feat_arr = group[features].values
            targ_arr = group[target].values
            if len(feat_arr) > size:
                for i in range(len(feat_arr) - size):
                    X.append(feat_arr[i : i + size])
                    y.append(targ_arr[i + size])
        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

    X_train, y_train = create_windows(train_df, window_size)
    X_test, y_test = create_windows(test_df, window_size)
    return X_train, y_train, X_test, y_test


class PhysicsInformedBMSNet(nn.Module):
    def __init__(self, input_dim):
        super(PhysicsInformedBMSNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=96, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


def train_physics_model():
    window_size = 35 
    X_train, y_train, X_test, y_test = prepare_soc_physics_features(window_size)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=512, shuffle=False)
    
    model = PhysicsInformedBMSNet(input_dim=4).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-3)
    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    epochs = 12
    print(f"🏋️ Training Physics-Informed Framework...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        model.eval()
        test_mae = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                test_mae += torch.sum(torch.abs(preds - batch_y)).item()
                
        scheduler.step()
        print(f"Epoch {epoch+1} -> Loss: {train_loss/len(X_train):.6f} | B0007 MAE: {(test_mae/len(X_test))*100:.2f}%")
        
    torch.save(model.state_dict(), PROCESSED_DIR / "high_perf_soc_model.pt")
    print("Physics anchored model saved over legacy weights.")

if __name__ == "__main__":
    train_physics_model()