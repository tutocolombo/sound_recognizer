"""Utility functions"""
import base64
import mimetypes
from pathlib import Path
from typing import Union

import miniaudio
import numpy as np


def to_categorical(y, num_classes):
    """1-hot encode a tensor."""
    return np.eye(num_classes, dtype="uint8")[y]


def read_audio_file(
    audio_file: Union[Path, str, bytes],
    n_channels=1,
    output_format=miniaudio.SampleFormat.SIGNED32,
):
    """Read and decode an audio file, whether it's the raw bytes or a path."""
    if isinstance(audio_file, bytes):
        _decode = miniaudio.decode
    else:
        _decode = miniaudio.decode_file

    return _decode(audio_file, nchannels=n_channels, output_format=output_format)


def get_b64_encoded_data_uri(audio_file: Union[Path, str]):
    mime_type, _ = mimetypes.guess_type(audio_file)
    encoded_audio = encode_b64_audio(audio_file)
    return f"data:{mime_type};base64,{encoded_audio.decode()}"


def encode_b64_audio(audio_file: Union[Path, str]):
    """Encode a file as a base64 string."""
    with open(audio_file, "rb") as a:
        audio_b = a.read()
    encoded_audio = base64.b64encode(audio_b)
    return encoded_audio


def read_b64_audio_file(data_uri: str):
    """Load base64-encoded audio files."""
    try:
        return read_b64_string(data_uri)
    except Exception as exception:
        raise ValueError(
            "Could not load audio from b64 {}: {}".format(data_uri, exception)
        ) from exception


def read_b64_string(b64_string, return_data_type=False):
    """Read a base64-encoded string"""
    data_header, b64_data = split_and_validate_b64_string(b64_string)
    decoded_audio = base64.b64decode(b64_data)
    if return_data_type:
        return data_header, decoded_audio
    else:
        return decoded_audio


def split_and_validate_b64_string(b64_string: str):
    """Return the data_type and data of a b64 string, with validation."""
    header, data = b64_string.split(",", 1)
    assert header.startswith("data:")
    assert header.endswith(";base64")
    data_header = header.split(";")[0].split(":")[1].split("/")
    return data_header, data
