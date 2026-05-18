import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob


class MultiFolderMultiCSVDataset(Dataset):
    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='M', data=None, target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None, **kwargs):

        # Initialize parameters
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # Handle size parameter
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 24
        else:
            self.seq_len = size[0]  # input_len
            self.label_len = size[1]  # label_len
            self.pred_len = size[2]  # pred_len
        print("size:", size)
        # Extract additional parameters from kwargs
        self.task_name = kwargs.get('task_name', 'finetune')
        self.use_norm = kwargs.get('use_norm', True)
        self.downstream_task = kwargs.get('downstream_task', 'forecast')
        self.model_id = kwargs.get('model_id', '')

        print(f"Initializing dataset with seq_len={self.seq_len}, label_len={self.label_len}, pred_len={self.pred_len}")
        print(f"Task: {self.task_name}, Downstream: {self.downstream_task}")

        self.__read_data__()

    def __read_data__(self):
        # Build full data path
        full_data_path = os.path.join(self.root_path, self.data_path)

        if not os.path.exists(full_data_path):
            raise ValueError(f"Data path {full_data_path} does not exist")

        # Get all sub-folders (expected 4 sub-folders)
        sub_folders = [f for f in os.listdir(full_data_path)
                       if os.path.isdir(os.path.join(full_data_path, f))]
        sub_folders.sort()

        print(f"Found {len(sub_folders)} sub-folders in {full_data_path}")

        # Store all samples
        self.samples = []  # Each sample has shape [seq_len, 4]
        self.sample_names = []

        # Get all CSV files from each sub-folder, ensuring equal counts
        csv_files_by_folder = {}
        max_files = 0

        for folder_name in sub_folders:
            folder_path = os.path.join(full_data_path, folder_name)
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
            csv_files.sort()
            csv_files_by_folder[folder_name] = csv_files
            max_files = max(max_files, len(csv_files))
            print(f"Folder {folder_name}: {len(csv_files)} CSV files")

        # Expect exactly 4 sub-folders
        if len(sub_folders) != 4:
            print(f"Warning: Expected 4 sub-folders, found {len(sub_folders)}")

        # Combine data: each sample consists of CSV files at the same index across the 4 folders
        for i in range(max_files):
            sample_features = []
            valid_sample = True

            for folder_name in sub_folders:
                if i < len(csv_files_by_folder[folder_name]):
                    csv_file = csv_files_by_folder[folder_name][i]
                    try:
                        df = pd.read_csv(csv_file)
                        # Assume the second column contains numeric data
                        if len(df.columns) >= 2:
                            values = df.iloc[:, 1].values.astype(float)
                            sample_features.append(values)
                        else:
                            valid_sample = False
                            break
                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")
                        valid_sample = False
                        break
                else:
                    valid_sample = False
                    break

            if valid_sample and len(sample_features) == 4:
                # Transpose to [seq_len, 4]
                sample_array = np.array(sample_features).T
                self.samples.append(sample_array)
                self.sample_names.append(f"sample_{i}")

        print(f"Total samples loaded: {len(self.samples)}")

        if len(self.samples) == 0:
            raise ValueError("No valid samples loaded")

        # Normalize each feature dimension independently
        if self.use_norm:
            # Collect all sample data for computing mean and std
            all_data = np.concatenate([sample for sample in self.samples], axis=0)  # [total_timesteps, 4]
            self.mean = np.mean(all_data, axis=0)  # [4]
            self.std = np.std(all_data, axis=0)  # [4]

            # Normalize each sample
            for i in range(len(self.samples)):
                self.samples[i] = (self.samples[i] - self.mean) / (self.std + 1e-8)
            print("Applied normalization to each feature dimension")

        # Split dataset by sample
        total_samples = len(self.samples)

        # Adjust split ratios based on task
        if self.task_name == 'pretrain':
            train_ratio, val_ratio = 0.8, 0.2  # Pre-training does not need a test set
        else:
            train_ratio, val_ratio = 0.7, 0.2

        if self.flag == 'train':
            start_idx = 0
            end_idx = int(total_samples * train_ratio)
        elif self.flag == 'val':
            start_idx = int(total_samples * train_ratio)
            end_idx = int(total_samples * (train_ratio + val_ratio))
        else:  # test
            start_idx = int(total_samples * (train_ratio + val_ratio))
            end_idx = total_samples

        self.samples = self.samples[start_idx:end_idx]
        self.sample_names = self.sample_names[start_idx:end_idx]

        print(f"{self.flag} set: {len(self.samples)} samples, range: [{start_idx}, {end_idx})")
        print(f"Sample shape: {self.samples[0].shape}")  # Should be [seq_len, 4]

    def _create_time_mark(self, length):
        """Create time markers."""
        return torch.zeros((length, 1))

    def __getitem__(self, index):
        sample_data = self.samples[index]  # Shape: [seq_len, num_features]

        # For pre-training: reconstruction task
        if self.task_name == 'pretrain':
            total_len = sample_data.shape[0]

            if total_len > self.seq_len:
                # Randomly select a starting position
                max_start = total_len - self.seq_len
                start_idx = np.random.randint(0, max_start)
                seq_x = sample_data[start_idx:start_idx + self.seq_len]
            else:
                # If sample is too short, repeat-pad
                repeats = (self.seq_len // total_len) + 1
                padded_data = np.tile(sample_data, (repeats, 1))
                seq_x = padded_data[:self.seq_len]

            seq_y = seq_x  # Pre-training target is the input itself

            # Create time markers
            seq_x_mark = self._create_time_mark(seq_x.shape[0])
            seq_y_mark = self._create_time_mark(seq_y.shape[0])

            return torch.from_numpy(seq_x).float(), torch.from_numpy(seq_y).float(), seq_x_mark, seq_y_mark
        else:
            # Fine-tuning: predict future sequence
            total_len = sample_data.shape[0]

            # Ensure sequence is long enough
            if total_len < self.seq_len + self.pred_len:
                # If too short, repeat-pad
                repeats = (self.seq_len + self.pred_len) // total_len + 1
                sample_data = np.tile(sample_data, (repeats, 1))
                total_len = sample_data.shape[0]

            # Randomly select a starting position
            max_start = total_len - self.seq_len - self.pred_len
            if max_start > 0:
                start_idx = np.random.randint(0, max_start)
            else:
                start_idx = 0

            s_begin = start_idx
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_x = sample_data[s_begin:s_end]
            seq_y = sample_data[r_begin:r_end]

            # Create time markers
            seq_x_mark = self._create_time_mark(seq_x.shape[0])
            seq_y_mark = self._create_time_mark(seq_y.shape[0])

            return torch.from_numpy(seq_x).float(), torch.from_numpy(seq_y).float(), seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.samples)


def multi_folder_csv_data_provider(args, flag):
    # Use the same size parameter format as other datasets
    size = [args.seq_len, args.label_len, args.pred_len]

    # Adjust parameters based on task type
    if getattr(args, 'task_name', 'finetune') == 'pretrain':
        # For pre-training, set pred_len and label_len to 0 (input and output have the same length)
        size[1] = 0  # label_len
        size[2] = 0  # pred_len

    # Pass all required parameters
    kwargs = {
        'task_name': getattr(args, 'task_name', 'finetune'),
        'use_norm': getattr(args, 'use_norm', True),
        'downstream_task': getattr(args, 'downstream_task', 'forecast'),
        'model_id': getattr(args, 'model_id', '')
    }

    data_set = MultiFolderMultiCSVDataset(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=size,
        features=args.features,
        target=args.target,
        scale=args.use_norm,
        timeenc=0 if getattr(args, 'embed', 'timeF') != 'timeF' else 1,
        freq=args.freq,
        **kwargs
    )

    # Adjust batch_size based on dataset size
    batch_size = args.batch_size
    if len(data_set) < batch_size:
        batch_size = max(1, len(data_set) // 2)
        print(f"Adjusting batch_size to {batch_size} for {flag} set with {len(data_set)} samples")

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=(flag == 'train'),
        num_workers=getattr(args, 'num_workers', 0),
        drop_last=(flag == 'train'),
        pin_memory=True
    )

    print(f"{flag} dataset: {len(data_set)} samples, data loader: {len(data_loader)} batches")
    return data_set, data_loader
