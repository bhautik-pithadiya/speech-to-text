import os
import wget
import json
import shutil
from faster_whisper import WhisperModel
import whisperx
import torch
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
from src import *
import time
from datetime import timedelta
from numba import cuda
import contractions
# Name of the audio file ---> Change it to folder path containing multiple audio files.
#audio_path = "/home/ksuser/LS/APAK.ai-main/audio_files/1696528455061_1000085836312_1012_2224792.mp3"

torch.set_num_threads(6)

# Whether to enable music removal from speech, helps increase diarization quality but uses alot of ram
enable_stemming = False

# (choose from 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large')
whisper_model_name = "medium.en"

# replaces numerical digits with their pronounciation, increases diarization accuracy 
suppress_numerals = True 

#models

def init_models():
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    whisper_model = WhisperModel(whisper_model_name, device="cpu", compute_type="int8")
    msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to("cpu")
    punct_model = PunctuationModel(model="kredor/punctuate-all")
    return whisper_model, msdd_model, punct_model 


def process(audio_path, whisper_model, msdd_model, punct_model):
    
#    audio_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".mp3") or f.endswith(".wav")]

    startTime = time.time()
    # if enable_stemming:
    #     # Isolate vocals from the rest of the audio

    #     return_code = os.system(
    #         f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "temp_outputs"'
    #     )

    #     if return_code != 0:
    #         logging.warning("Source splitting failed, using original audio file.")
    #         vocal_target = audio_path
    #     else:
    #         vocal_target = os.path.join(
    #             "temp_outputs", "htdemucs", os.path.basename(audio_path[:-4]), "vocals.wav"
    #         )
    # else:
    
    vocal_target = audio_path
    
    # Run on GPU with FP16
    # whisper_model = WhisperModel(whisper_model_name, device="cuda", compute_type="int8")
    # or run on GPU with INT8
    # whisper_model = WhisperModel(whisper_model_name, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # whisper_model = WhisperModel(whisper_model_name, device="cpu", compute_type="int16")
    if suppress_numerals:
        numeral_symbol_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    else:
        numeral_symbol_tokens = None

    segments, info = whisper_model.transcribe(
        vocal_target,
        beam_size=5,
        word_timestamps=True,
        suppress_tokens=numeral_symbol_tokens,
        vad_filter=True,
    )
    whisper_results = []
    toal_info = []
    for segment in segments:
        whisper_results.append(segment._asdict())
        
    # clear gpu vram
    # del whisper_model
    # torch.cuda.empty_cache()

    if info.language in wav2vec2_langs:
        device = "cpu"
        alignment_model, metadata = whisperx.load_align_model(
            language_code=info.language, device=device
        )
        result_aligned = whisperx.align(
            whisper_results, alignment_model, metadata, vocal_target, device
        )
        word_timestamps = filter_missing_timestamps(result_aligned["word_segments"])

        # clear gpu vram
        # del alignment_model
        # torch.cuda.empty_cache()
    else:
        word_timestamps = []
        for segment in whisper_results:
            for word in segment["words"]:
                word_timestamps.append({"word": word[2], "start": word[0], "end": word[1]})


    sound = AudioSegment.from_file(vocal_target).set_channels(1)
    
    ROOT = os.getcwd()
    temp_path = os.path.join(ROOT, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)
    sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")
    os.system( f'cp /home/ksuser/noise/whisper-diarization/nemo_msdd_configs/diar_infer_telephonic.yaml "{temp_path}"')

    # Initialize NeMo MSDD diarization model
    # msdd_model = NeuralDiarizer(cfg=create_config(temp_path))

    msdd_model.diarize()
    
    
    # del msdd_model
    # torch.cuda.empty_cache()
    # Reading timestamps <> Speaker Labels mapping

    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    
    
    if info.language in punct_model_langs:
        # restoring punctuation in the transcript to help realign the sentences
        # punct_model = PunctuationModel(model="kredor/punctuate-all")
        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = punct_model.predict(words_list)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    else:
        print(
            f'Punctuation restoration is not available for {whisper_results["language"]} language.'
        )

    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)
    # print(ssm)
    # print(get_speaker_aware_transcript(ssm,))
    # final_output = []
    # for details in ssm:
    #     final_output.append(f"{details['speaker']} : {details['text']}")
    final_string = ''
    previous = ''
    for details in ssm:
        curr = details['speaker']
        if previous != curr:
            final_string += details['speaker'] + ": " + details['text']
            # final_output.append(f"{details['speaker']} : {details['text']}")
        else:
            final_string += details['text']
        previous = curr
    expanded_string = []
    for word in final_string.split():
        expanded_string.append(contractions.fix(word))
    return " ".join(expanded_string)

    
    with open(f"{audio_path[:-4]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)
        
    # with open(f"{audio_path[:-4]}.srt", "w", encoding="utf-8-sig") as srt:
    #     write_srt(ssm, srt)
    
    endTime = time.time()
    print("Time taken:", str(timedelta(seconds=endTime - startTime)))
    
    # cleanup(temp_path)
    
    # del punct_model
    # torch.cuda.empty_cache()
    
