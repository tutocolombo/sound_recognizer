import torchaudio


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
