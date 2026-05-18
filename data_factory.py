from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Physio, Dataset_PEMS, Dataset_Epilepsy
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Electricity': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Weather': Dataset_Custom,
    'ECL': Dataset_Custom,
    'ILI': Dataset_Custom,
    'm4': None,  # removed: no longer needed for RAPID-Hydrology
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'HAR': Dataset_Physio,
    'EEG': Dataset_Physio,
    'PEMS03': Dataset_PEMS,
    'PEMS04': Dataset_PEMS,
    'PEMS07': Dataset_PEMS,
    'PEMS08': Dataset_PEMS,
    'Epilepsy': Dataset_Epilepsy,
}
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Dataset_DirectLoad(Dataset):
    def __init__(self, root_path, flag='train', input_len=None, pred_len=24,
                 features='M', data_path='', target='Pre', scale=True,
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Direct-load dataset for fixed-length samples.
        - Each CSV file serves as an independent sample
        - Directly slices the first input_len time steps as input, the next pred_len as target
        - No sliding window needed
        """
        self.input_len = input_len if input_len is not None else 96
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.scale = scale

        # Dataset split
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # Read all CSV files
        data_files = []
        data_dir = os.path.join(self.root_path, self.data_path)

        if os.path.isdir(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    data_files.append(os.path.join(data_dir, file))
        else:
            data_files = [data_dir]

        print(f"Found {len(data_files)} data files")

        all_samples = []
        valid_files = []

        for file_path in data_files:
            try:
                # Read CSV, skip header, extract numeric data only
                df = pd.read_csv(file_path, header=None, skiprows=1)

                # Check if data length is sufficient
                if len(df) >= self.input_len + self.pred_len:
                    # Extract the last four feature columns (assumed: Pre, Tm, Win, Rhu)
                    data = df.iloc[:self.input_len + self.pred_len, -4:].values.astype(np.float32)
                    all_samples.append(data)
                    valid_files.append(file_path)
                else:
                    print(f"File {file_path}: insufficient data length: {len(df)} < {self.input_len + self.pred_len}")

            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

        if not all_samples:
            raise ValueError("No valid data files found")

        self.all_samples = np.array(all_samples)
        print(f"Data shape: {self.all_samples.shape}")  # [num_samples, total_len, num_features]

        # Split dataset
        num_samples = len(self.all_samples)
        train_size = int(num_samples * self.train_ratio)
        val_size = int(num_samples * self.val_ratio)
        test_size = num_samples - train_size - val_size

        indices = np.random.permutation(num_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        if self.set_type == 0:  # train
            self.samples = self.all_samples[train_indices]
        elif self.set_type == 1:  # val
            self.samples = self.all_samples[val_indices]
        else:  # test
            self.samples = self.all_samples[test_indices]

        print(f"{['train', 'val', 'test'][self.set_type]} set samples: {len(self.samples)}")

        # Normalization
        if self.scale:
            # Fit scaler on training set only
            if self.set_type == 0:  # train
                train_data_reshaped = self.samples.reshape(-1, self.samples.shape[-1])
                self.scaler.fit(train_data_reshaped)
            else:
                # For val/test, use scaler fitted on training data
                train_data = self.all_samples[train_indices].reshape(-1, self.all_samples.shape[-1])
                self.scaler.fit(train_data)

            original_shape = self.samples.shape
            self.samples = self.samples.reshape(-1, original_shape[-1])
            self.samples = self.scaler.transform(self.samples)
            self.samples = self.samples.reshape(original_shape)

    def __getitem__(self, index):
        sample_data = self.samples[index]  # Shape: [input_len + pred_len, num_features]

        # Directly split into input and output
        seq_x = sample_data[:self.input_len]  # Input sequence: [input_len, num_features]
        seq_y = sample_data[self.input_len:self.input_len + self.pred_len]  # Target sequence: [pred_len, num_features]

        # Feature selection
        if self.features == 'S':
            # Univariate: use only target feature
            target_idx = 0  # Assume first feature 'Pre' is the target
            seq_x = seq_x[:, target_idx:target_idx + 1]
            seq_y = seq_y[:, target_idx:target_idx + 1]
        elif self.features == 'MS':
            # Multivariate input, univariate output: predict only the first feature
            seq_y = seq_y[:, 0:1]

        # Create time markers (simple version)
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.samples)

    def inverse_transform(self, data):
        """Inverse-transform normalized data."""
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data


class Dataset_DirectLoad_Pretrain(Dataset):
    """
    Direct-load dataset for pre-training.
    """

    def __init__(self, root_path, data_path='', input_len=96, features='M',
                 scale=True, mask_ratio=0.15, train_ratio=0.8):

        self.input_len = input_len
        self.mask_ratio = mask_ratio
        self.features = features
        self.scale = scale
        self.train_ratio = train_ratio

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # Read data files
        data_files = []
        data_dir = os.path.join(self.root_path, self.data_path)

        if os.path.isdir(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    data_files.append(os.path.join(data_dir, file))
        else:
            data_files = [data_dir]

        print(f"Pre-train: found {len(data_files)} data files")

        all_samples = []

        for file_path in data_files:
            try:
                # Skip header, read numeric data directly
                df = pd.read_csv(file_path, header=None, skiprows=1)

                if len(df) >= self.input_len:
                    # Extract the last four feature columns
                    data = df.iloc[:self.input_len, -4:].values.astype(np.float32)
                    all_samples.append(data)

            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

        if not all_samples:
            raise ValueError("No valid data files found")

        self.all_samples = np.array(all_samples)
        print(f"Pre-train data shape: {self.all_samples.shape}")

        # Pre-train uses training portion only
        train_size = int(len(self.all_samples) * self.train_ratio)
        self.samples = self.all_samples[:train_size]

        # Normalization
        if self.scale:
            original_shape = self.samples.shape
            data_reshaped = self.samples.reshape(-1, original_shape[-1])
            self.scaler.fit(data_reshaped)

            self.samples = self.samples.reshape(-1, original_shape[-1])
            self.samples = self.scaler.transform(self.samples)
            self.samples = self.samples.reshape(original_shape)

    def __getitem__(self, index):
        sample_data = self.samples[index]  # Shape: [input_len, num_features]

        # For pre-training, input and output are identical (reconstruction task)
        seq_x = sample_data
        seq_y = sample_data.copy()

        # Feature selection
        if self.features == 'S':
            seq_x = seq_x[:, 0:1]
            seq_y = seq_y[:, 0:1]

        # Create time markers
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.samples)


def data_provider(args, flag):
    """
    Direct-load data provider function.
    """
    # Check whether pre-training or fine-tuning
    if hasattr(args, 'task_name') and args.task_name == 'pretrain':
        # Pre-training data
        Data = Dataset_DirectLoad_Pretrain
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            input_len=args.input_len,  # Use seq_len as input_len
            features=args.features,
            scale=args.use_norm,
            mask_ratio=getattr(args, 'mask_ratio', 0.15),
            train_ratio=0.8
        )
    else:
        # Fine-tuning data
        Data = Dataset_DirectLoad
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            input_len=args.input_len,  # Use seq_len as input_len
            pred_len=args.pred_len,
            features=args.features,
            target=args.target,
            scale=args.use_norm,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

    print(f"{flag} dataset size: {len(data_set)}")

    # Adjust batch_size based on stage
    if flag == 'train':
        batch_size = args.batch_size
    else:
        batch_size = min(args.batch_size, 32)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True if flag == 'train' else False,
        num_workers=args.num_workers,
        drop_last=True if flag == 'train' else False
    )

    return data_set, data_loader
