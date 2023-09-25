import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from datasets import load_dataset
import numpy as np
import os
import json

def calculate_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))

    for i in range(len(ref_words) + 1):
        d[i, 0] = i

    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):

            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:

                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)
    return d[len(ref_words), len(hyp_words)] , len(ref_words)
    

#model = Speech2TextForConditionalGeneration.from_pretrained("s2t-small-librispeech-asr")
#processor = Speech2TextProcessor.from_pretrained("s2t-small-librispeech-asr")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
model.config.forced_decoder_ids = None

ds = load_dataset(
    "LibriSpeech",
    "clean",
    split="validation"
)
print(ds)

for filename in os.listdir('GBLyrics'):
    if filename == '101935856_1430128.json':
        file_path = os.path.join('GBLyrics', filename)
        # Open and read the JSON file
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            break

error=0
total=0

for num in range(20):
    print(num)
    print(ds[num])
    
    if len(ds[num]["audio"]["array"]) != 0:
        input_features = processor(
            ds[num]["audio"]["array"],
            sampling_rate=16_000,
            return_tensors="pt"
        ).input_features  # Batch size 1
    else:
        continue



    generated_ids = model.generate(input_features=input_features)

    print(generated_ids)

    #transcription = processor.batch_decode(generated_ids)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

    print(transcription)
    
    #new_t = transcription[0].split('</s>')
    new_t = transcription[0].split('.')

    number_index=ds[num]['audio']['path'].split('.')[0].split('_')[-1]
    print('index:')
    print(number_index)

    e,t=calculate_wer(data[int(number_index)-1]['l'], new_t[0])
    print(data[int(number_index)-1]['l'])
    print(new_t[0])

    error=error+e
    total=total+t

    print(error/total)
    print(error)
    print(total)
    


