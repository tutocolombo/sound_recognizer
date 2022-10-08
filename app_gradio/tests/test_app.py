import json
import os

import requests

from app_gradio import app
from sound_recognizer import util


os.environ["CUDA_VISIBLE_DEVICES"] = ""


TEST_SOUND = "sound_recognizer/tests/support/sounds/sample_birds.wav"


def test_local_run():
    """A quick test to make sure we can build the app and ping the API locally."""
    backend = app.PredictorBackend()
    frontend = app.make_frontend(fn=backend.run)

    # run the UI without blocking
    frontend.launch(share=False, prevent_thread_lock=True)
    local_url = frontend.local_url
    get_response = requests.get(local_url)
    assert get_response.status_code == 200

    local_api = f"{local_url}api/predict"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"data": util.get_b64_encoded_data_uri(TEST_SOUND)})
    post_response = requests.post(local_api, data=payload, headers=headers)
    assert "error" not in post_response.json()
    assert "data" in post_response.json()
