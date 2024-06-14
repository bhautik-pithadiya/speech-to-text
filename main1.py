from src import diarize
from src import *
import time
from datetime import timedelta
# "/home/ksuser/Documents/conversationTranscribe/data/input_audio/1696528084008_1000022968016_1022_2224792_100.mp3"


path = "./1696528151059_1000050599709_1028_2224792.mp3"

# startTime = time.time()
# whisper_model, msdd_model, punct_model = diarize.init_models()
# endTime = time.time()
# print("Model Loaded in:", str(timedelta(seconds=endTime - startTime)))


# choice = "y"
# while(choice =="y"):
#     # path = input("Enter path:")
#     path = '/home/ksuser/Documents/conversationTranscribe/data/input_audio/1696528084008_1000022968016_1022_2224792_100.mp3'
#     startTime = time.time()
#     transcript = diarize.process(path,whisper_model,msdd_model,punct_model)
#     endTime = time.time()
#     print("Transcription done in:", str(timedelta(seconds=endTime - startTime)))
#     # print(transcript)
#     choice = input("cpntinue? :")
    



# For Parallel computing
startTime = time.time()
whisper_model,msdd, punct_model = diarize.init_models()
endTime = time.time()
print("Model Loaded in:", str(timedelta(seconds=endTime - startTime)))
choice = "y"
while(choice =="y"):
    # path = input("Enter path:")
    # path = '/1696528084008_1000022968016_1022_2224792_100.mp3'
    startTime = time.time()
    transcript = diarize.process(path,whisper_model,msdd,punct_model)
    endTime = time.time()
    print("Transcription done in:", str(timedelta(seconds=endTime - startTime)))
    print(transcript)
    choice = input("continue? :")    
