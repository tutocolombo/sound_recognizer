"""Detects sounds.

Example usage as a script:

  python sound_recognizer/environmental_sound_classifier.py
"""
import argparse
from pathlib import Path
from typing import Union

import torch

from sound_recognizer.data.esc50 import AudioToMelSpecDb
import sound_recognizer.util as util


STAGED_MODEL_DIRNAME = (
    Path(__file__).resolve().parent / "artifacts" / "sound-recognizer"
)
MODEL_FILE = "model.pt"


class EnvironmentalSoundClassifier:
    """Recognizes sounds from a recording"""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = STAGED_MODEL_DIRNAME / MODEL_FILE
        self.model = torch.jit.load(model_path)
        self.mapping = self.model.mapping
        self.transform = AudioToMelSpecDb()

    @torch.no_grad()
    def predict(self, audio_file: Union[str, Path, bytes]) -> str:
        """Predict/infer sounds in input audio file (which can be a file path or url, or the file bytes)."""
        audio = util.read_audio_file(audio_file)
        audio_tensor = torch.Tensor(audio.samples) / (1 << 31)  # Normalized tensor
        image_tensor = torch.Tensor(
            self.transform(audio_tensor).data
        )  # Spectrogram tensor
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # [ B, C, H, W ]
        logits = self.model(image_tensor)
        y_pred = torch.argmax(logits, dim=1)
        return self.mapping[y_pred[0]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "filename",
        type=str,
        help="Name for an audio file. This can be a local path, a URL, a URI from AWS/GCP/Azure storage, an HDFS path, or any other resource locator supported by the smart_open library.",
    )
    args = parser.parse_args()

    sound_recognizer = EnvironmentalSoundClassifier()
    pred_str = sound_recognizer.predict(args.filename)
    print(pred_str)


if __name__ == "__main__":
    main()
