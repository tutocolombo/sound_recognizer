"""Metadata for the MNIST dataset."""
import sound_recognizer.metadata.shared as shared

DOWNLOADED_DATA_DIRNAME = shared.DOWNLOADED_DATA_DIRNAME
BASE_FOLDER = "ESC-50-master"
URL = "https://codeload.github.com/karolpiczak/ESC-50/zip/master"
ZIP_FILENAME = "ESC-50-master.zip"
ZIP_MD5 = "d5c1e28f805d90cbca24a6d9fbbebe2b"
NUM_FILES_IN_DIR = 2000
AUDIO_DIR = "audio"
TARGET_COL = "target"
LABEL_COL = "category"
FILE_COL = "filename"
META_FILENAME = "meta/esc50.csv"
META_MD5 = "54a0d0055a10bb7df84ad340a148722e"

DIMS = (1, 128, 431)
OUTPUT_DIMS = (1,)
MAPPING = [
    "dog",
    "rooster",
    "pig",
    "cow",
    "frog",
    "cat",
    "hen",
    "insects",
    "sheep",
    "crow",
    "rain",
    "sea_waves",
    "crackling_fire",
    "crickets",
    "chirping_birds",
    "water_drops",
    "wind",
    "pouring_water",
    "toilet_flush",
    "thunderstorm",
    "crying_baby",
    "sneezing",
    "clapping",
    "breathing",
    "coughing",
    "footsteps",
    "laughing",
    "brushing_teeth",
    "snoring",
    "drinking_sipping",
    "door_wood_knock",
    "mouse_click",
    "keyboard_typing",
    "door_wood_creaks",
    "can_opening",
    "washing_machine",
    "vacuum_cleaner",
    "clock_alarm",
    "clock_tick",
    "glass_breaking",
    "helicopter",
    "chainsaw",
    "siren",
    "car_horn",
    "engine",
    "train",
    "church_bells",
    "airplane",
    "fireworks",
    "hand_saw",
]
