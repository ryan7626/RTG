import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class GestureDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = df.iloc[:, :-1].values.astype('float32')
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(df.iloc[:, -1])
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = self.y[idx]
        return x, y

    def get_label_encoder(self):
        return self.le
