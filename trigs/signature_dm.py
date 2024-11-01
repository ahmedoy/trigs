from trigs.signature_dataset import SignatureDataset
from torchvision import transforms as T
from lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
import torch


class SignatureDataModule(LightningDataModule):
    def __init__(self, data_path, dataset_name, baseline, opt_mode, split_rand_seed=43,
                 train_ds_use_frac=1.0) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.baseline = baseline
        self.data_path = data_path
        self.opt_mode = opt_mode
        self.split_rand_seed = split_rand_seed
        self.train_ds_use_frac = train_ds_use_frac
        self.train_loader = self.valid_loader = self.test_loader = None

    def setup(self, stage=None):

        if self.dataset_name.startswith("TIN") or self.dataset_name.startswith("ImageNet"):
            transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif self.dataset_name == "CIFAR10" or self.dataset_name.startswith("TrojAI"):
            transform = T.ToTensor()
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

        clean_dataset = SignatureDataset(
            base_path=self.data_path,
            dataset_name=self.dataset_name,
            mode_="train",
            transform=transform,
            model_type_="clean",
            baseline=self.baseline,
            opt_mode=self.opt_mode
        )
        poisoned_dataset = SignatureDataset(
            base_path=self.data_path,
            dataset_name=self.dataset_name,
            mode_="train",
            transform=transform,
            model_type_="poisoned",
            baseline=self.baseline,
            opt_mode=self.opt_mode
        )
        if self.train_ds_use_frac < 1.0:
            clean_use = int(len(clean_dataset) * self.train_ds_use_frac)
            clean_dataset, _ = random_split(clean_dataset, (clean_use, len(clean_dataset) - clean_use),
                                            generator=torch.Generator().manual_seed(self.split_rand_seed))
            poisoned_use = int(len(poisoned_dataset) * self.train_ds_use_frac)
            poisoned_dataset, _ = random_split(poisoned_dataset, (poisoned_use, len(poisoned_dataset) - poisoned_use),
                                               generator=torch.Generator().manual_seed(self.split_rand_seed))
        clean_val_len = int(len(clean_dataset) * 0.1)
        valid_clean_dataset, train_clean_dataset = random_split(
            clean_dataset,
            [clean_val_len, len(clean_dataset) - clean_val_len],
            generator=torch.Generator().manual_seed(self.split_rand_seed),
        )

        poisoned_val_len = int(len(poisoned_dataset) * 0.1)
        valid_poisoned_dataset, train_poisoned_dataset = random_split(
            poisoned_dataset,
            [poisoned_val_len, len(poisoned_dataset) - poisoned_val_len],
            generator=torch.Generator().manual_seed(self.split_rand_seed),
        )

        valid_dataset = torch.utils.data.ConcatDataset(
            [valid_clean_dataset, valid_poisoned_dataset]
        )
        train_dataset = torch.utils.data.ConcatDataset(
            [train_clean_dataset, train_poisoned_dataset]
        )
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(valid_dataset)}")
        self.valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=2)
        self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

        test_clean_dataset = SignatureDataset(
            base_path=self.data_path,
            dataset_name=self.dataset_name,
            mode_="test",
            transform=transform,
            model_type_="clean",
            baseline=self.baseline,
            opt_mode=self.opt_mode
        )
        test_poisoned_dataset = SignatureDataset(
            base_path=self.data_path,
            dataset_name=self.dataset_name,
            mode_="test",
            transform=transform,
            model_type_="poisoned",
            baseline=self.baseline,
            opt_mode=self.opt_mode
        )
        test_dataset = torch.utils.data.ConcatDataset(
            [test_clean_dataset, test_poisoned_dataset]
        )
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_loader
