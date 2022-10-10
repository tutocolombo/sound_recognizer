from datetime import datetime
import os
from typing import List, Optional

import gradio as gr
from gradio.components import Component

from app_gradio import s3_util
from sound_recognizer.util import read_b64_string


class SoundClassificationS3Logger(gr.FlaggingCallback):
    """A FlaggingCallback that logs flagged sound classification data to S3."""

    def setup(self, components: List[Component], flagging_dir: str):
        """Sets up the SoundClassificationS3Logger by creating or attaching to an S3 Bucket."""
        self._counter = 0
        bucket_name = s3_util.make_unique_bucket_name(prefix=flagging_dir, seed="")
        self.bucket = s3_util.get_or_create_bucket(bucket_name)
        s3_util.enable_bucket_versioning(self.bucket)
        # s3_util.add_access_policy(self.bucket)
        (
            self.audio_component_idx,
            self.label_component_idx,
        ) = find_audio_and_label_components(components)

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None) -> int:
        """Sends flagged outputs and feedback and audio inputs to S3."""
        audio = flag_data[self.audio_component_idx]["data"]
        classification = flag_data[self.label_component_idx]["label"]
        data_type, audio_bytes = read_b64_string(audio, return_data_type=True)
        feedback = f'{datetime.now().strftime("%Y%m%d.%H%M%S.%f")}_{flag_option}_{classification.replace(" ","_")}_'
        s3_util.to_s3(self.bucket, audio_bytes, feedback=feedback, filetype=data_type)
        self._counter += 1

        return self._counter


def find_audio_and_label_components(components: List[Component]):
    audio_component_idx, label_component_idx = None, None

    for idx, component in enumerate(components):
        if isinstance(component, (gr.inputs.Audio, gr.components.Audio)):
            audio_component_idx = idx
        elif isinstance(component, (gr.outputs.Label, gr.components.Label)):
            label_component_idx = idx
    if audio_component_idx is None:
        raise RuntimeError(
            f"No audio input found in gradio interface with components {components}"
        )
    elif label_component_idx is None:
        raise RuntimeError(
            f"No label output found in gradio interface with components {components}"
        )

    return audio_component_idx, label_component_idx


def get_api_key() -> Optional[str]:
    """Convenience method for fetching the Gantry API key."""
    api_key = os.environ.get("GANTRY_API_KEY")
    return api_key
