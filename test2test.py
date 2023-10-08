import torch
import torch.nn as nn
import torch.optim as optim
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

# Define a CNN model
class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        # Define your CNN layers here, e.g., convolutional layers, pooling, etc.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        # Add more layers as needed

    def forward(self, x):
        # Define the forward pass of your CNN
        x = self.conv1(x)
        # Add more layers and operations as needed
        return x

# Create a custom model that combines the CNN and Whisper models
class CombinedModel(nn.Module):
    def __init__(self, cnn_model, whisper_model,forced_decoder_ids):
        super(CombinedModel, self).__init__()
        self.cnn = cnn_model
        self.whisper = whisper_model

    def forward(self, x):
        x = self.cnn(x)  # Pass input through the CNN
        x = whisper_model.generate(input_features=x,forced_decoder_ids=forced_decoder_ids)  # Pass the output through the Whisper model
        x = processor.batch_decode(x, skip_special_tokens=True)
        return x

# Load the Whisper processor and model
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

# Load the LibriSpeech validation dataset
ds = load_dataset("LibriSpeech", "clean", split="validation")

# Define your CNN hyperparameters
#in_channels = 1  # Number of input channels (adjust as needed)
#out_channels = 80  # Number of output channels (adjust as needed)
#kernel_size = 3  # Convolutional kernel size (adjust as needed)

# Instantiate the CNN model
cnn_model = SpectrogramCNN()

# Create the combined model with the CNN and frozen Whisper model
combined_model = CombinedModel(cnn_model, whisper_model,forced_decoder_ids)


learning_rate=0.01
# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)  # Adjust learning rate as needed

num_epochs=1
print_interval=1
# Training loop
for epoch in range(num_epochs):
    for num in range(len(ds)):

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

        optimizer.zero_grad()

        # Pass input features through the combined model
        outputs = combined_model(input_features)

        print(outputs)

        new_t = outputs[0].split('.')
        number_index=ds[num]['audio']['path'].split('.')[0].split('_')[-1]
        file_name = ds[num]['audio']['path'].split('\\')[7].split('.')[0].split('_')

        for filename in os.listdir('GBLyrics'):
            if filename == file_name[0]+'_'+file_name[1]+'.json':
                file_path = os.path.join('GBLyrics', filename)
                # Open and read the JSON file
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    break
        e,t=calculate_wer(data[int(number_index)-1]['l'], new_t[0])

        # Calculate loss and backpropagate
        loss = e/t  # Adjust labels as needed
        loss_tensor = torch.tensor(loss, dtype=torch.float32)  # Convert to PyTorch tensor
        loss_tensor.backward()
        optimizer.step()

        # Print training statistics (optional)
        if (num + 1) % print_interval == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {num + 1}/{len(ds)}, Loss: {loss.item()}")

        break

# Save your trained CNN model if needed
#torch.save(cnn_model.state_dict(), "cnn_model.pth")
