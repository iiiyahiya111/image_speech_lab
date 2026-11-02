"""
    pip install SpeechRecognition
    python -m pip install --only-binary=:all: pocketsphinx
"""
import speech_recognition as sr

recognizer = sr.Recognizer()

# 1. Load the audio file
with sr.AudioFile('audio.wav') as source:
    recognizer.adjust_for_ambient_noise(source, duration=0.2)  # estimate noise
    audio = recognizer.record(source)                          # read whole file

# 2. Recognize offline with PocketSphinx
try:
    text = recognizer.recognize_sphinx(audio)
    print('Recognized text:')
    print(text)
except sr.UnknownValueError:
    print('Speech was not understood (try a clearer recording).')
except sr.RequestError as e:
    print('PocketSphinx error (is pocketsphinx installed?):', e)