"""Loading a dataset into memory and accessing data."""

from abc import ABC, abstractmethod
import os
from pathlib import Path
import shutil
import tarfile
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wget
from loguru import logger

from globalanchors.anchor_types import Dataset
from globalanchors.constants import DATA_PATH


class DataLoader(ABC):
    def __init__(self, test_split: float = 0.2, val_split: float = 0.1):
        self.test_split = test_split
        self.val_split = val_split
        self.dataset_path = self.download_dataset()
        self.dataset = self.load_dataset()

    @abstractmethod
    def download_dataset(self) -> Path:
        return NotImplementedError

    def load_dataset(self) -> Dataset:
        data = []
        labels = []
        f_names = ["0.txt", "1.txt"]
        logger.info("Loading dataset...")
        for l, f in enumerate(tqdm(f_names)):
            for line in open(self.dataset_path / f, "rb"):
                try:
                    line.decode("utf-8")
                except:
                    continue
                data.append(line.strip())
                labels.append(l)
        train, test, train_labels, test_labels = train_test_split(
            data, labels, test_size=self.test_split
        )
        train, val, train_labels, val_labels = train_test_split(
            train, train_labels, test_size=self.val_split
        )
        return {
            "train_data": train,
            "train_labels": train_labels,
            "val_data": val,
            "val_labels": val_labels,
            "test_data": test,
            "test_labels": test_labels,
        }


class PolarityDataLoader(DataLoader):
    # override
    def download_dataset(self) -> Path:
        data_dir = Path(DATA_PATH)
        data_dir.mkdir(parents=True, exist_ok=True)
        # download dataset and extract
        URL = "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"
        if not os.path.exists(data_dir / "PolarityData"):
            logger.info("Downloading polarity dataset...")
            wget.download(URL, str(data_dir / "rt-polaritydata.tar.gz"))
            logger.info("Unzipping polarity dataset...")
            with tarfile.open(data_dir / "rt-polaritydata.tar.gz") as tar:
                tar.extractall(data_dir)
        else:
            logger.info("Found existing polarity dataset.")
            return data_dir / "PolarityData"
        ## standardize format for loading
        # create necessary files and directories
        positive_file = data_dir / "PolarityData" / "1.txt"
        positive_file.parent.mkdir(parents=True, exist_ok=True)
        negative_file = data_dir / "PolarityData" / "0.txt"
        negative_file.parent.mkdir(parents=True, exist_ok=True)
        # copy data to new files
        with open(positive_file, "wb") as f:
            with open(
                data_dir / "rt-polaritydata/rt-polarity.pos", "rb"
            ) as pos:
                f.write(pos.read())
        with open(negative_file, "wb") as f:
            with open(
                data_dir / "rt-polaritydata/rt-polarity.neg", "rb"
            ) as neg:
                f.write(neg.read())
        # clean-up
        (data_dir / "rt-polaritydata.tar.gz").unlink()
        (data_dir / "rt-polaritydata.README.1.0.txt").unlink()
        shutil.rmtree(data_dir / "rt-polaritydata")
        return data_dir / "PolarityData"


class SubjectivityDataLoader(DataLoader):
    # override
    def download_dataset(self) -> Path:
        data_dir = Path(DATA_PATH)
        data_dir.mkdir(parents=True, exist_ok=True)
        # download dataset and extract
        URL = "http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz"
        if not os.path.exists(data_dir / "SubjectivityData"):
            logger.info("Downloading subjectivity dataset...")
            wget.download(URL, str(data_dir / "rotten_imdb.tar.gz"))
            logger.info("Unzipping subjectivity dataset...")
            with tarfile.open(data_dir / "rotten_imdb.tar.gz") as tar:
                tar.extractall(data_dir)
        else:
            logger.info("Found existing subjectivity dataset.")
            return data_dir / "SubjectivityData"
        ## standardize format for loading
        # create necessary files and directories
        subjective_file = data_dir / "SubjectivityData" / "1.txt"
        subjective_file.parent.mkdir(parents=True, exist_ok=True)
        objective_file = data_dir / "SubjectivityData" / "0.txt"
        objective_file.parent.mkdir(parents=True, exist_ok=True)
        # copy data to new files
        with open(subjective_file, "wb") as f:
            with open(data_dir / "quote.tok.gt9.5000", "rb") as subj:
                f.write(subj.read())
        with open(objective_file, "wb") as f:
            with open(data_dir / "plot.tok.gt9.5000", "rb") as obj:
                f.write(obj.read())
        # clean-up
        (data_dir / "rotten_imdb.tar.gz").unlink()
        (data_dir / "plot.tok.gt9.5000").unlink()
        (data_dir / "quote.tok.gt9.5000").unlink()
        (data_dir / "subjdata.README.1.0").unlink()
        return data_dir / "PolarityData"
