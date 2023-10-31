from pydub import AudioSegment

# Load the input audio file
input_audio = AudioSegment.from_file("audio.wav")

# Pitch up the audio by an octave (12 semitones)
pitched_audio = input_audio.speedup(playback_speed=2)

# Export the modified audio to a new file
pitched_audio.export("audio2.wav", format="wav")
