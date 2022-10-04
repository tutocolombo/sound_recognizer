"""Detects sounds.

Example usage as a script:

  python sound_recognizer/environmental_sound_classifier.py
"""
import argparse
from pathlib import Path
from typing import Union

import torch
import torchaudio

from sound_recognizer.data.esc50 import AudioToMelSpecDb


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
    def predict(self, audio: Union[str, Path]) -> str:
        """Predict/infer sounds in input audio file (which can be a file path or url)."""
        wav, _ = torchaudio.load(audio)
        image_tensor = torch.Tensor(self.transform(wav).data)
        if len(image_tensor) > 1:
            # If the image has more that 1 channel, we average the channels but keep the shape
            image_tensor = torch.mean(image_tensor, 0).unsqueeze(0)
        image_tensor = image_tensor.unsqueeze(0)
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

    text_recognizer = EnvironmentalSoundClassifier()
    pred_str = text_recognizer.predict(args.filename)
    print(pred_str)


if __name__ == "__main__":
    main()
