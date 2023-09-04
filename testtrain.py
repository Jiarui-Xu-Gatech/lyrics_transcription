import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset

model = Speech2TextForConditionalGeneration.from_pretrained("s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("s2t-small-librispeech-asr")

ds = load_dataset(
    "LibriSpeech",
    "clean",
    split="validation"
)
print(ds)

input_features = processor(
    ds[2]["audio"]["array"],
    sampling_rate=16_000,
    return_tensors="pt"
).input_features  # Batch size 1

print(input_features)

generated_ids = model.generate(input_features=input_features)

print(generated_ids)

transcription = processor.batch_decode(generated_ids)

print(transcription)
