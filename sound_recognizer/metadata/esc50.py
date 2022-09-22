"""Metadata for the MNIST dataset."""
import sound_recognizer.metadata.shared as shared

DOWNLOADED_DATA_DIRNAME = shared.DOWNLOADED_DATA_DIRNAME

DIMS = 1
OUTPUT_DIMS = (1,)
MAPPING = list(range(50))

TRAIN_SIZE = 1280
VAL_SIZE = 320

BASE_FOLDER = 'ESC-50-master'
URL = "https://codeload.github.com/karolpiczak/ESC-50/zip/master"
ZIP_FILENAME = "ESC-50-master.zip"
ZIP_MD5 = 'd5c1e28f805d90cbca24a6d9fbbebe2b'
NUM_FILES_IN_DIR = 2000
AUDIO_DIR = 'audio'
LABEL_COL = 'category'
FILE_COL = 'filename'
META_FILENAME = "meta/esc50.csv"
META_MD5 = '54a0d0055a10bb7df84ad340a148722e'
