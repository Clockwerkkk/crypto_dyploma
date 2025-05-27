from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch

class MultiStepDataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, seq_len=120):
        self.seq_len = seq_len
        df = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
        self.scaler = StandardScaler()
        features = df[feature_cols].values
        self.features = self.scaler.fit_transform(features)
        self.targets = df[target_cols].values

    def __len__(self):
        return len(self.targets) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)