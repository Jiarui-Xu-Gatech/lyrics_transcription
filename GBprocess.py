import os
import json
from pydub import AudioSegment
from moviepy.editor import AudioFileClip
import pydub
import ffmpeg


def split_m4a_to_wav(m4a_file,name, destination_folder,start_time,end_time):
    audio = AudioSegment.from_file(m4a_file, format="mp3")
    #audio = AudioSegment.from_wav(m4a_file)

    # Split the audio into parts
    part = audio[start_time:end_time]

    # Save each part as a WAV file
    wav_filename = os.path.join(destination_folder, name+'.wav')
    part.export(wav_filename, format="wav")

# Specify the folder path where your JSON files are located
folder_path = 'GBLyrics'

source_folder = 'GBVocals'
destination_folder = 'fragments'

source_out = 'GBMp3'

# Create a list to store the extracted data
'''
for audioname in os.listdir(source_folder):
    audiopoint = audioname.split('.')
    m4a_file = os.path.join(source_folder, audioname)
    input_m4a_file = m4a_file
    output_mp3_file = os.path.join(source_out,audiopoint[0]+'.mp3')

    # Load the M4A file
    audio_clip = AudioFileClip(input_m4a_file)

    # Convert and write to WAV file
    audio_clip.write_audiofile(output_mp3_file, codec='mp3')
    

'''

# Loop through all files in the folder
contin = False
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        print(filename)
        # Open and read the JSON file
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            filename_frag = filename.split('.')
            # You can access specific data elements from the JSON here
            # For example, if your JSON has a 'name' key, you can extract it like this:
            # name = data['name']
            
            # Append the data to the list

            

            
            if filename_frag[0] == '403519875_2804378':
                contin = True
            if contin == True:
                for audioname in os.listdir(source_out):
                    audiopoint = audioname.split('.')

                    if audiopoint[-1] != 'wav' and audiopoint[-1] != 'm4a' and audiopoint[-1] != 'mp3':
                        continue
                    audioname_frag = audioname.split('-')
                    
                    m4a_file = os.path.join(source_out, audioname)

                    #
                    #split_m4a_to_wav(m4a_file,'test',destination_folder,data[0]['t']*1000,data[1]['t']*1000)
                    #

                    if audioname_frag[0] == filename_frag[0] and audioname_frag[3] == 'F':
                        print(m4a_file)
                        
                        for c in range(len(data)-1):
                            split_m4a_to_wav(m4a_file,filename_frag[0]+'_'+str(c+1),destination_folder,data[c]['t']*1000,data[c+1]['t']*1000)
                            
                
        
                    
            
                
            
           

# Now, you have a list containing data from all the JSON files in the folder
# You can process this data further as needed


