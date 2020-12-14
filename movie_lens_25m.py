import csv
import hashlib
import logging
import pandas as pd
import zipfile
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl

class MovieLens25M(torch.utils.data.Dataset):
    def __init__(self, data: pd.core.frame.DataFrame):
        self._data = data
        self.n_users = self._data["userId"].nunique()
        self.n_movies = self._data["movieId"].nunique()
        self.max_user_idx = self._data["userId"].max()
        self.max_movie_idx = self._data["movieId"].max()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        user_id = self._data.loc[index, "userId"]
        movie_id = self._data.loc[index, "movieId"]
        rating = self._data.loc[index, "rating"]

        return torch.tensor([user_id, movie_id], dtype=torch.long), torch.tensor(rating)

class MovieLens25MDataModule(pl.LightningDataModule):

    URL = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"
    MD5 = "6b51fb2759a8657d3bfcbfc42b592ada"
    ML_DIR_NAME = "ml-25m"

    def __init__(self, data_dir: str = './', batch_size: int = 64):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size

    def prepare_data(self):
        zip_path = self.data_dir / Path(self.URL).name

        # directory already exists, skip
        ml_25m_dir = self.data_dir / self.ML_DIR_NAME
        if ml_25m_dir.exists():
            logging.info(f"{ml_25m_dir} exists, skipping download")
            return

        # download
        if zip_path.exists():
            logging.info(f"{zip_path} exists, skipping download")
        else:
            logging.info(f"downloading {self.URL}")
            with urllib.request.urlopen(self.URL) as r:
                with open(zip_path, "wb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        f.write(chunk)

        # verify MD5
        md5 = hashlib.md5()
        with open(zip_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)

        if md5.hexdigest() != self.MD5:
            err = f"{zip_path} MD5 wrong, expected {self.MD5}, got {md5.hexdigest()}"
            logging.critical(err)
            raise ValueError(err)
        else:
            logging.info(f"{zip_path} MD5 check passed")

        # extract data
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(self.data_dir)

    def setup(self, stage=None, head=None):
        data = pd.read_csv(
            self.data_dir / self.ML_DIR_NAME / "ratings.csv",
            dtype={"userId": np.int64, "movieId": np.int64, "rating": np.float32, "timestamp": np.int64}
        )
        if head:
            data = data.head(head)

        largest = (
            data.groupby("userId", group_keys=False)
                .apply(lambda x: x.nlargest(2, columns=["timestamp"]))
        )

        test_idx_set = set(largest.iloc[::2].index.values)
        valid_idx_set = set(largest.iloc[1::2].index.values)

        # find index of latest rating by each user, put in set
        test_idxs = data.index.isin(test_idx_set)
        valid_idxs = data.index.isin(valid_idx_set)
        train_idxs = ~(test_idxs | valid_idxs)

        if stage == "fit" or stage is None:
            train = data.iloc[train_idxs].sample(frac=1, random_state=42).reset_index(drop=True)
            self.train = MovieLens25M(train)

            valid = data.iloc[valid_idxs].reset_index(drop=True)
            self.valid = MovieLens25M(valid)

        if stage == "test" or stage is None:
            test = data.iloc[test_idxs].reset_index(drop=True)
            self.test = MovieLens25M(test)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train, self.batch_size, num_workers=6
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid, self.batch_size, num_workers=6
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, self.batch_size)
