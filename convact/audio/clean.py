import librosa
from defaults import RATE


def clean_audio(data):
    new, _ = librosa.effects.trim(
        y=data,
        frame_length=RATE,
        top_db=40
    )
    return new
