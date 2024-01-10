import random

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader


class flDPnnDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers, seed) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        train_data = open('data/flDPnn/flDPnn_Training_Annotation.txt').read().splitlines()[10:]
        val_data = open('data/flDPnn/flDPnn_Validation_Annotation.txt').read().splitlines()[10:]
        test_data = open('data/flDPnn/flDPnn_DissimiTest_Annotation.txt').read().splitlines()[10:]
        self.train_set = [train_data[i:i + 3] for i in range(0, len(train_data), 7)]
        self.val_set = [val_data[i:i + 3] for i in range(0, len(val_data), 7)]
        self.test_set = [test_data[i:i + 3] for i in range(0, len(test_data), 7)]

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
            collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=self.collate)

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=self.collate)

    def collate(self, batch):
        return batch


class ApodDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        train_data = open('data/DFL/DFL_benchmark_TR166/TR133.txt').read().splitlines()
        val_data = open('data/DFL/DFL_benchmark_TR166/DV33.txt').read().splitlines()
        te64_data = open('data/DFL/TE64.txt').read().splitlines()
        te82_data = open('data/DFL/TE82.txt').read().splitlines()
        self.train_set = [train_data[i:i + 3] for i in range(0, len(train_data), 3)]
        self.val_set = [val_data[i:i + 3] for i in range(0, len(val_data), 3)]
        self.te64_set = [te64_data[i:i + 3] for i in range(0, len(te64_data), 3)]
        self.te82_set = [te82_data[i:i + 3] for i in range(0, len(te82_data), 3)]

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
            collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=self.collate)

    def test_dataloader(self, te='te64'):
        return DataLoader(self.te64_set if te == 'te64' else self.te82_set, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, collate_fn=self.collate)

    def collate(self, batch):
        return batch


class DisoLipPredDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        data = open('data/DisoLipPred/TrainingDataset_Server.txt').read().splitlines()
        train_data = data[15:441]
        val_data = data[445:]
        test_data = open('data/DisoLipPred/TestDataset_Server.txt').read().splitlines()[10:]
        self.train_set = [train_data[i:i + 3] for i in range(0, len(train_data), 3)]
        self.val_set = [val_data[i:i + 3] for i in range(0, len(val_data), 3)]
        self.test_set = [test_data[i:i + 3] for i in range(0, len(test_data), 3)]

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
            collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=self.collate)

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=self.collate)

    def collate(self, batch):
        return batch


class DeepDisoBindDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        data = open('data/DeepDISOBind/TrainingDataset.txt').read().splitlines()[8:]
        train_data, val_data = data[:1428], data[1428:]
        test_data = open('data/DeepDISOBind/TestDataset.txt').read().splitlines()[8:]
        self.train_set = [train_data[i:i + 6] for i in range(0, len(train_data), 6)]
        self.val_set = [val_data[i:i + 6] for i in range(0, len(val_data), 6)]
        self.test_set = [test_data[i:i + 6] for i in range(0, len(test_data), 6)]

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
            collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=self.collate)

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=self.collate)

    def collate(self, batch):
        return batch


class NetSurfP30DataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        data = open('data/NetSurfP-3.0/Train_HHblits.txt').read().splitlines()
        train_data, val_data = data[:-500 * 3], data[-500 * 3:]
        casp12_data = open('data/NetSurfP-3.0/CASP12_HHblits.txt').read().splitlines()
        cb513_data = open('data/NetSurfP-3.0/CB513_HHblits.txt').read().splitlines()
        ts115_data = open('data/NetSurfP-3.0/TS115_HHblits.txt').read().splitlines()
        self.train_set = [train_data[i:i + 3] for i in range(0, len(train_data), 3)]
        self.val_set = [val_data[i:i + 3] for i in range(0, len(val_data), 3)]
        self.casp12_set = [casp12_data[i:i + 3] for i in range(0, len(casp12_data), 3)]
        self.cb513_set = [cb513_data[i:i + 3] for i in range(0, len(cb513_data), 3)]
        self.ts115_set = [ts115_data[i:i + 3] for i in range(0, len(ts115_data), 3)]

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
            collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=self.collate)

    def test_dataloader(self, t='casp12'):
        d = self.casp12_set if t == 'casp12' else self.cb513_set if t == 'cb513' else self.ts115_set
        return DataLoader(
            d, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate)

    def collate(self, batch):
        return batch


class Dm4229DataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_data = open('data/DM4229/train.txt').read().splitlines()
        val_data = open('data/DM4229/val.txt').read().splitlines()

        self.train_set = [train_data[i:i + 3] for i in range(0, len(train_data), 3)]
        self.val_set = [val_data[i:i + 3] for i in range(0, len(val_data), 3)]

    def train_dataloader(self, exclude=None):
        if exclude is None:
            return DataLoader(
                self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                collate_fn=self.collate)
        return DataLoader(
            [d for d in self.train_set if d[0][1:] not in exclude],
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
            collate_fn=self.collate)

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=self.collate)

    def collate(self, batch):
        return batch


class UnionIdpDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_data = open('data/Train/train.txt').read().splitlines()
        val_data = open('data/Train/val.txt').read().splitlines()

        self.train_set = [train_data[i:i + 3] for i in range(0, len(train_data), 3)]
        self.val_set = [val_data[i:i + 3] for i in range(0, len(val_data), 3)]

    def train_dataloader(self, exclude=None):
        if exclude is None:
            return DataLoader(
                self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
                collate_fn=self.collate)

        print(len(self.train_set), len([d for d in self.train_set if d[0][1:] not in exclude]))
        return DataLoader(
            [d for d in self.train_set if d[0][1:] not in exclude],
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True,
            collate_fn=self.collate)

    def val_dataloader(self, exclude=None):
        if exclude is None:
            return DataLoader(
                self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                collate_fn=self.collate)
        return DataLoader(
            [d for d in self.val_set if d[0][1:] not in exclude],
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=self.collate)

    def collate(self, batch):
        return batch


class VariousIdpTestDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        casp_data = open('data/Test/CASP_clean.txt').read().splitlines()
        disorder723_data = open('data/Test/DISORDER723_clean.txt').read().splitlines()
        mxd494_data = open('data/Test/MXD494_clean.txt').read().splitlines()
        sl329_data = open('data/Test/SL329_clean.txt').read().splitlines()

        self.casp_set = [casp_data[i:i + 3] for i in range(0, len(casp_data), 3)]
        self.disorder723_set = [disorder723_data[i:i + 3] for i in range(0, len(disorder723_data), 3)]
        self.mxd494_set = [mxd494_data[i:i + 3] for i in range(0, len(mxd494_data), 3)]
        self.sl329_set = [sl329_data[i:i + 3] for i in range(0, len(sl329_data), 3)]

    def test_dataloader(self, t='casp'):
        d = self.casp_set if t == 'casp' else self.disorder723_set if t == 'disorder723' else self.mxd494_set if t == 'mxd494' else self.sl329_set
        return DataLoader(
            d, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.collate)

    def collate(self, batch):
        return batch


class DisProt2022DecDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        disprot_data = open('data/DisProt_clean.txt').read().splitlines()
        self.disprot_set = [disprot_data[i:i + 3] for i in range(0, len(disprot_data), 3)]

    def test_dataloader(self):
        return DataLoader(
            self.disprot_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=self.collate)

    def collate(self, batch):
        return batch


class CaidDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # caid_data = open('data/CAID_clean.txt').read().splitlines()
        caid_data = open('data/CAID.txt').read().splitlines()
        self.caid_set = [caid_data[i:i + 3] for i in range(0, len(caid_data), 3)]

    def test_dataloader(self):
        return DataLoader(
            self.caid_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=self.collate)

    def collate(self, batch):
        return batch


class Caid2DataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # caid_data = open('data/CAID_clean.txt').read().splitlines()
        caid_data = open('data/CAID2/disorder_pdb_2.txt').read().splitlines()
        self.caid_set = [caid_data[i:i + 3] for i in range(0, len(caid_data), 3)]

    def test_dataloader(self):
        return DataLoader(
            self.caid_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
            collate_fn=self.collate)

    def collate(self, batch):
        return batch


if __name__ == '__main__':
    dm = UnionIdpDataModule(1, 24)
    ids = open('data/Train/exclude.casp.txt').read().splitlines()
    print(ids)
    for batch in dm.train_dataloader(ids):
        print(batch)
        break
