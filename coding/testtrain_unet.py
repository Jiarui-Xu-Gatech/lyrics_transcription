import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from datasets import load_dataset
import numpy as np
import os
import json

import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


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
#processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
#model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
#model.config.forced_decoder_ids = None
#processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
#model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")



ds = load_dataset(
    "LibriSpeech",
    "clean",
    split="validation"
)
print(ds)

#for filename in os.listdir('GBLyrics'):
#    if filename == '101935856_1430128.json':
#        file_path = os.path.join('GBLyrics', filename)
#        # Open and read the JSON file
#        with open(file_path, 'r') as json_file:
#            data = json.load(json_file)
#            break

error=0
total=0

data=0

for num in range(2000):
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

    



    generated_ids = model.generate(input_features=input_features,forced_decoder_ids=forced_decoder_ids)

    print(generated_ids)

    #transcription = processor.batch_decode(generated_ids)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)

    print(transcription)
    
    #new_t = transcription[0].split('</s>')
    new_t = transcription[0].split('.')

    number_index=ds[num]['audio']['path'].split('.')[0].split('_')[-1]
    print('index:')
    print(number_index)

    file_name = ds[num]['audio']['path'].split('\\')[7].split('.')[0].split('_')

    

    for filename in os.listdir('GBLyrics'):
        if filename == file_name[0]+'_'+file_name[1]+'.json':
            file_path = os.path.join('GBLyrics', filename)
            # Open and read the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                break
    e,t=calculate_wer(data[int(number_index)-1]['l'], new_t[0])
    print(data[int(number_index)-1]['l'])
    print(new_t[0])

    error2=error+e
    total2=total+t

    if e < 2*t:
        error = error2
        total = total2
    else:
        print("Unstable happened!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print(error/total)
    print(error)
    print(total)

    


