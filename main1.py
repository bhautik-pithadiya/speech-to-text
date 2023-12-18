from src import diarize
from src import *

path = "/home/ksuser/Downloads/Speech_to_text_api/1696528151059_1000050599709_1028_2224792.mp3"

whisper_model, msdd_model, punct_model = diarize.init_models()
transcript = diarize.process(path,whisper_model,msdd_model,punct_model)
print(transcript)
