import lightning as L
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import torchvision
import torchmetrics

import timm

class Food101Dataset(Dataset):
    def __init__(self, ds, augment=True):
        super().__init__()

        self.ds = ds

        if augment:
            trafos = [v2.RandomResizedCrop(size=(224, 224)),
                      v2.RandomHorizontalFlip(0.5),
                      v2.RandomVerticalFlip(0.5),
                      v2.ColorJitter(),
                      v2.PILToTensor(),
                      v2.ToDtype(torch.float32, scale=True),
                      v2.Normalize(mean=[0.458, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                     ]
        else:
            trafos = [v2.Resize(size=(224, 224)),
                      v2.PILToTensor(),
                      v2.ToDtype(torch.float32, scale=True),
                      v2.Normalize(mean=[0.458, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                      ]
        self.transform = v2.Compose(trafos)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        sample, label = self.ds[idx]
        sample = self.transform(sample)
        return sample, label

class Food101DataModule(L.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=8):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers


    def prepare_data(self, split='train'):
        '''Download the dataset'''
        self.ds = torchvision.datasets.Food101(download=True, root='./data', split=split)

    def setup(self, stage='train'):
        '''Setup the dataset for the given split'''
        if stage == 'train':
            gen = torch.Generator().manual_seed(456123)
            split1, split2 = torch.utils.data.random_split(self.ds, [0.8, 0.2], generator=gen)
            self.train_ds = Food101Dataset(split1, augment=True)
            self.val_ds = Food101Dataset(split2, augment=False)
        elif stage == 'test':
            self.test_ds = Food101Dataset(self.ds, augment=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

class Food101Model(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model('resnet18', pretrained=True)
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_metric = torchmetrics.Accuracy(task='multiclass', num_classes=1000)


    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.backbone(x)
        loss = self.loss_fn(yhat, y)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.backbone(x)
        loss = self.loss_fn(yhat, y)
        self.val_metric(yhat, y)
        self.log('val/loss', loss, sync_dist=True)
        self.log('val/accuracy', self.val_metric, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.backbone(x)
        loss = self.loss_fn(yhat, y)
        self.val_metric(yhat, y)
        self.log('test/loss', loss, sync_dist=True)
        self.log('test/accuracy', self.val_metric, sync_dist=True)
        return yhat

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

