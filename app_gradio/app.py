"""Provide a recording of environmental sound and get back the classification!"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable

import gradio as gr
import miniaudio
import requests

from app_gradio.flagging import SoundClassificationS3Logger
from sound_recognizer.environmental_sound_classifier import EnvironmentalSoundClassifier
import sound_recognizer.util as util

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # do not use GPU

logging.basicConfig(level=logging.INFO)

APP_DIR = Path(__file__).resolve().parent  # what is the directory for this application?
FAVICON = str(
    APP_DIR / "sound_icon.png"
)  # path to a small image for display in browser tab and social media
README = APP_DIR / "README.md"  # path to an app readme file in HTML/markdown

DEFAULT_PORT = 11700


def main(args):
    predictor = PredictorBackend(url=args.model_url)
    frontend = make_frontend(
        fn=predictor.run,
        flagging=args.flagging,
    )
    frontend.launch(
        server_name="0.0.0.0",  # make server accessible, binding all interfaces  # noqa: S104
        server_port=args.port,  # set a port to bind to, failing if unavailable
        favicon_path=FAVICON,  # what icon should we display in the address bar?
        share=args.share,
    )


def make_frontend(
    fn: Callable[[str], str],
    flagging: bool = False,
):
    """Creates a gradio.Interface frontend for an audio classification function."""
    examples_dir = Path("sound_recognizer") / "tests" / "support" / "sounds"
    example_fnames = [
        elem
        for elem in os.listdir(examples_dir)
        if elem.endswith(".wav") or elem.endswith(".mp3")
    ]
    example_paths = [examples_dir / fname for fname in example_fnames]

    examples = [[str(path)] for path in example_paths]

    if flagging:
        allow_flagging = "manual"
        # callback for logging input audio, output text, and feedback to s3
        flagging_callback = SoundClassificationS3Logger()
    else:
        allow_flagging = "never"
        flagging_callback = None

    readme = _load_readme(with_logging=allow_flagging == "manual")

    # build a basic browser interface to a Python function
    frontend = gr.Interface(
        fn=fn,  # which Python function are we interacting with?
        outputs=gr.components.Label(
            value="Silence"
        ),  # what output widgets does it need? the default Label widget
        # what input widgets does it need? we configure an audio widget
        inputs=gr.components.Audio(type="filepath", label="Recorded Audio"),
        title="Sound Recognizer",  # what should we display at the top of the page?
        thumbnail=FAVICON,  # what should we display when the link is shared, e.g. on social media?
        description=__doc__,  # what should we display just above the interface?
        article=readme,  # what long-form content should we display below the interface?
        examples=examples,  # which potential inputs should we provide?
        cache_examples=False,  # should we cache those inputs for faster inference? slows down start
        allow_flagging=allow_flagging,  # should we show users the option to "flag" outputs?
        flagging_options=[
            "incorrect",
            "other",
        ],  # what options do users have for feedback?
        flagging_callback=flagging_callback,
        flagging_dir="sound-recognizer-flagged-samples",
    )

    return frontend


class PredictorBackend:
    """Interface to a backend that serves predictions.

    To communicate with a backend accessible via a URL, provide the url kwarg.

    Otherwise, runs a predictor locally.
    """

    def __init__(self, url=None):
        if url is not None:
            print("Using endpoint predictor")
            self.url = url
            self._predict = self._predict_from_endpoint
        else:
            model = EnvironmentalSoundClassifier()
            self._predict = model.predict

    def run(self, path):
        pred, metrics = self._predict_with_metrics(path)
        _log_inference(pred, metrics)
        return pred.capitalize().replace("_", " ")

    def _predict_with_metrics(self, audio_file):
        try:
            file_info = miniaudio.get_file_info(audio_file)
            pred = self._predict(audio_file)
            metrics = {
                "n_channels": file_info.nchannels,
                "sample_rate": file_info.sample_rate,
                "num_frames": file_info.num_frames,
                "duration": file_info.duration,
                "file_format": file_info.file_format,
            }
        except miniaudio.DecodeError as e:
            pred, metrics = str(e), {"error": repr(e)}
        return pred, metrics

    def _predict_from_endpoint(self, audio):
        """Send an audio file to an endpoint that accepts JSON and return the predicted class.

        The endpoint should expect a base64 representation of the file, encoded as a string,
        under the key "audio". It should return the predicted class under the key "pred".

        Parameters
        ----------
        audio
            An audio recording of some sound to be converted into a string.

        Returns
        -------
        pred
            A string containing the predictor's guess of the sound.
        """
        headers = {"Content-type": "application/json"}
        payload = json.dumps({"audio": util.get_b64_encoded_data_uri(audio)})

        response = requests.post(self.url, data=payload, headers=headers)
        pred = response.json()["pred"]

        return pred


def _log_inference(pred, metrics):
    for key, value in metrics.items():
        logging.info(f"METRIC {key} {value}")
    logging.info(f"PRED >begin\n{pred}\nPRED >end")


def _load_readme(with_logging=False):
    with open(README) as f:
        lines = f.readlines()
        if not with_logging:
            lines = lines[: lines.index("<!-- logging content below -->\n")]

        readme = "".join(lines)
    return readme


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_url",
        default=None,
        type=str,
        help="Identifies a URL to which to send image data. Data is base64-encoded, converted to a utf-8 string, and then set via a POST request as JSON with the key 'image'. Default is None, which instead sends the data to a model running locally.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help=f"Port on which to expose this server. Default is {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--flagging",
        action="store_true",
        help="Pass this flag to allow users to 'flag' model behavior and provide feedback.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Pass this flag to create a temporary url with Gradio",
    )

    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    main(args)
