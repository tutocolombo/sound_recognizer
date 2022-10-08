"""AWS Lambda function serving text_recognizer predictions."""
import json

from sound_recognizer.environmental_sound_classifier import EnvironmentalSoundClassifier
import sound_recognizer.util as util

model = EnvironmentalSoundClassifier()


def handler(event, _context):
    """Provide main prediction API."""
    print("INFO loading audio file")
    audio = _load_audio(event)
    if audio is None:
        return {"statusCode": 400, "message": "audio not found in event"}
    print("INFO audio file loaded")
    print("INFO starting inference")
    pred = model.predict(audio)
    print("INFO inference complete")
    print("METRIC pred_length {}".format(len(pred)))
    print("INFO pred {}".format(pred))
    return {"pred": str(pred)}


def _load_audio(event):
    event = _from_string(event)
    event = _from_string(event.get("body", event))
    audio = event.get("audio")
    if audio is not None:
        print("INFO reading audio from event")
        return util.read_b64_audio_file(audio)
    else:
        return None


def _from_string(event):
    if isinstance(event, str):
        return json.loads(event)
    else:
        return event
