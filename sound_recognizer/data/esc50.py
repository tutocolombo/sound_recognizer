import argparse
import os
from typing import Callable, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
import torchaudio
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from sound_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
import sound_recognizer.metadata.esc50 as metadata


class ESC50DS(Dataset):
    """ESC50 Dataset. Same function as other torchaudio datasets"""

    def __init__(
        self,
        root: str,
        download: bool = False,
        train: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.train = train
        self._load_meta()

        self.transform = transform
        self.data, self.targets = self._load_data()

    def _load_meta(self):
        path = os.path.join(self.root, metadata.BASE_FOLDER, metadata.META_FILENAME)
        if not check_integrity(path, metadata.META_MD5):
            raise RuntimeError(
                "Dataset metadata file not found or corrupted."
                + " You can use download=True to download it"
            )

        data = pd.read_csv(path)
        index = data["fold"] != 5 if self.train else data["fold"] == 5
        self.df = data[index]
        self.class_labels = (
            data.sort_values(metadata.TARGET_COL)[metadata.LABEL_COL]
            .drop_duplicates()
            .to_list()
        )
        self.class_to_idx = {v: k for k, v in enumerate(self.class_labels)}

    def _load_data(self):
        data = []
        targets = []
        for _, row in self.df.iterrows():

            file_path = os.path.join(
                self.root,
                metadata.BASE_FOLDER,
                metadata.AUDIO_DIR,
                row[metadata.FILE_COL],
            )
            wav, sr = torchaudio.load(file_path)
            wav = wav if not self.transform else torch.Tensor(self.transform(wav).data)

            data.append(wav)
            targets.append(row[metadata.TARGET_COL])

        return data, targets

    def __getitem__(self, index):
        """
        Args
        ----
            index (int): Index

        Returns
        -------
            tuple: (audio, target) where target is index of the target class.
        """
        data, target = self.data[index], self.targets[index]

        return data, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        path = os.path.join(self.root, metadata.BASE_FOLDER, metadata.META_FILENAME)
        if not check_integrity(path, metadata.META_MD5):
            return False
        path = os.path.join(self.root, metadata.BASE_FOLDER, metadata.AUDIO_DIR)
        if len(next(os.walk(path))[2]) != metadata.NUM_FILES_IN_DIR:
            return False
        return True

    def download(self):
        if self._check_integrity():
            # print('Files already downloaded and verified')
            return

        download_and_extract_archive(
            url=metadata.URL,
            download_root=self.root,
            filename=metadata.ZIP_FILENAME,
            md5=metadata.ZIP_MD5,
        )

    def split_by_fold(self, fold=4):
        # fmt: off
        return Subset(self, self.df.index[self.df["fold"] != fold]), \
               Subset(self, self.df.index[self.df["fold"] == fold])
        # fmt: on


class ESC50(BaseDataModule):
    """ESC50 DataModule."""

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__(args)
        self.data_dir = metadata.DOWNLOADED_DATA_DIRNAME
        self.transform = AudioToMelSpecDb()
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS
        self.mapping = metadata.MAPPING

    def prepare_data(self, *args, **kwargs) -> None:
        ESC50DS(self.data_dir, download=True)

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        if stage == "fit" or stage is None:
            train_full = ESC50DS(self.data_dir, train=True, transform=self.transform)
            self.data_train, self.data_val = train_full.split_by_fold()  # type: ignore
        if stage == "test" or stage is None:
            self.data_test = ESC50DS(
                self.data_dir, train=False, transform=self.transform
            )


class AudioToMelSpecDb:
    """Transform to get a Mel Spectrogram in dB scale from an audio tensor"""

    def __init__(self):
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=20,
            f_max=8300,
        )
        self.mel_spectrogram_db_transform = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, audio):
        mel_spectrogram = self.mel_spectrogram_transform(audio)
        mel_spec_db = self.mel_spectrogram_db_transform(mel_spectrogram)
        return mel_spec_db


if __name__ == "__main__":
    load_and_print_info(ESC50)
