import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset
import numpy as np

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
    wer = d[len(ref_words), len(hyp_words)] / len(ref_words)
    return wer

model = Speech2TextForConditionalGeneration.from_pretrained("s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("s2t-small-librispeech-asr")

ds = load_dataset(
    "LibriSpeech",
    "clean",
    split="validation"
)
print(ds)

input_features = processor(
    ds[0]["audio"]["array"],
    sampling_rate=16_000,
    return_tensors="pt"
).input_features  # Batch size 1

print(input_features)

generated_ids = model.generate(input_features=input_features)

print(generated_ids)

transcription = processor.batch_decode(generated_ids)

print(transcription)
