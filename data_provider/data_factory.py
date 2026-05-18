import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Dataset_MyData(Dataset):
    def __init__(
        self,
        root_path,
        flag='train',
        input_len=2500,
        pred_len=400,
        features='M',
        data_path='',
        target='Pre',
        scale=True,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        step_size=100,
        scaler=None,
        test_no_sliding=True,
    ):
        """
        Time series forecasting dataset (no data leakage version).
        """
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.input_len = input_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.scale = scale
        self.step_size = step_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.test_no_sliding = test_no_sliding

        self.scaler = scaler if scaler is not None else StandardScaler()

        self.__read_data__()

    def __read_data__(self):
        # Read all CSV files
        data_files = []
        data_dir = os.path.join(self.root_path, self.data_path)
        if os.path.isdir(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith('.csv'):
                    data_files.append(os.path.join(data_dir, f))
        else:
            data_files = [data_dir]

        all_file_data = []
        for file in data_files:
            df = pd.read_csv(file, header=None, skiprows=1)
            data = df.iloc[:, -4:].values.astype(np.float32)
            all_file_data.append(data)

        # Build all sliding windows (in chronological order)
        windows = []
        for file_data in all_file_data:
            L = len(file_data)

            if self.set_type == 2 and self.test_no_sliding:
                # Test set: take only the last window (no information leakage)
                start = L - (self.input_len + self.pred_len)
                if start >= 0:
                    windows.append(file_data[start:start + self.input_len + self.pred_len])
            else:
                num = (L - self.input_len - self.pred_len) // self.step_size + 1
                for i in range(num):
                    s = i * self.step_size
                    windows.append(file_data[s:s + self.input_len + self.pred_len])

        windows = np.array(windows)

        # Chronological split (no shuffle)
        n = len(windows)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        if self.set_type == 0:
            self.samples = windows[:n_train]
        elif self.set_type == 1:
            self.samples = windows[n_train:n_train + n_val]
        else:
            self.samples = windows[n_train + n_val:]

        # Scaler fitted on train set only
        if self.scale:
            if self.set_type == 0:
                flat = self.samples.reshape(-1, self.samples.shape[-1])
                self.scaler.fit(flat)

            flat = self.samples.reshape(-1, self.samples.shape[-1])
            self.samples = self.scaler.transform(flat).reshape(self.samples.shape)

    def __getitem__(self, idx):
        data = self.samples[idx]
        seq_x = data[:self.input_len]
        seq_y = data[self.input_len:self.input_len + self.pred_len]

        if self.features == 'S':
            seq_x = seq_x[:, :1]
            seq_y = seq_y[:, :1]
        elif self.features == 'MS':
            seq_y = seq_y[:, :1]

        return (
            torch.FloatTensor(seq_x),
            torch.FloatTensor(seq_y),
            torch.zeros((seq_x.shape[0], 1)),
            torch.zeros((seq_y.shape[0], 1)),
        )

    def __len__(self):
        return len(self.samples)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_MyData_Pretrain(Dataset):
    def __init__(
        self,
        root_path,
        input_len=2500,
        features='M',
        data_path='',
        scale=True,
        train_ratio=0.8,
        step_size=100,
        scaler=None,
    ):
        self.root_path = root_path
        self.data_path = data_path
        self.input_len = input_len
        self.features = features
        self.scale = scale
        self.step_size = step_size
        self.train_ratio = train_ratio
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        data_files = []
        data_dir = os.path.join(self.root_path, self.data_path)
        if os.path.isdir(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith('.csv'):
                    data_files.append(os.path.join(data_dir, f))
        else:
            data_files = [data_dir]

        all_file_data = []
        for file in data_files:
            df = pd.read_csv(file, header=None, skiprows=1)
            data = df.iloc[:, -4:].values.astype(np.float32)
            all_file_data.append(data)

        windows = []
        for file_data in all_file_data:
            L = len(file_data)
            num = (L - self.input_len) // self.step_size + 1
            for i in range(num):
                s = i * self.step_size
                windows.append(file_data[s:s + self.input_len])

        windows = np.array(windows)
        n_train = int(len(windows) * self.train_ratio)
        self.samples = windows[:n_train]

        if self.scale:
            flat = self.samples.reshape(-1, self.samples.shape[-1])
            self.scaler.fit(flat)
            self.samples = self.scaler.transform(flat).reshape(self.samples.shape)

    def __getitem__(self, idx):
        data = self.samples[idx]
        seq_x = data[:self.input_len]
        if self.features == 'S':
            seq_x = seq_x[:, :1]
        return (
            torch.FloatTensor(seq_x),
            torch.FloatTensor(seq_x.copy()),
            torch.zeros((seq_x.shape[0], 1)),
            torch.zeros((seq_x.shape[0], 1)),
        )

    def __len__(self):
        return len(self.samples)


def data_provider(args, flag):
    if args.task_name == "pretrain":
        data_set = Dataset_MyData_Pretrain(
            root_path=args.root_path,
            data_path=args.data_path,
            input_len=args.input_len,
            features=args.features,
            scale=bool(args.use_norm),
            step_size=getattr(args, 'step_size', 100),
        )
    else:
        data_set = Dataset_MyData(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            input_len=args.input_len,
            pred_len=args.pred_len,
            features=args.features,
            target=args.target,
            scale=bool(args.use_norm),
            step_size=getattr(args, 'step_size', 100),
        )

    if flag == 'train':
        batch_size = args.batch_size
    else:
        batch_size = min(args.batch_size, 32)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True if flag == 'train' else False,
        num_workers=args.num_workers,
        drop_last=True if flag == 'train' else False,
    )

    return data_set, data_loader
