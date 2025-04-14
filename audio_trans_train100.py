import os
import librosa
import h5py
import numpy as np

processed_data=[]
root_folder="D:\\Anul 4\\LICENTA stuff\\baza de date\\clean\\train-clean-100"
file_count=0
batch=50
k=0

def save_to_h5(data):
    global file_count
    filename=f"train100_proc_data_{file_count}.h5"

    with h5py.File(filename, "w") as h5f:
        audio_group=h5f.create_group("audio_signals")
        transcript_group=h5f.create_group("transcriptions")

        for i, item in enumerate(data):
            audio_group.create_dataset(str(i), data=np.array(item['audio_signal'], dtype=np.float32))
            transcript_group.create_dataset(str(i), data=np.bytes_(item['transcription']))

    file_count+=1

for path, folders, files in os.walk(root_folder):

    audio_transcription={}

    for filename in files:
        if filename.endswith(".trans.txt"):
            text_file_path=os.path.join(path, filename)
            with open(text_file_path, 'r', encoding="utf-8") as f:
                for line in f:
                    audio_file, transcript=line.strip().split(' ', 1)
                    audio_transcription[audio_file]=transcript

    for audio_file, transcript in audio_transcription.items():
        # print(f"Procesare fisier: {audio_file}")
        try:
            audio_path=os.path.join(path, f"{audio_file}.flac")
            if os.path.exists(audio_path):
                audio, sr=librosa.load(audio_path, sr=16000)
                normalized_audio=librosa.util.normalize(audio)

                processed_data.append({
                    'audio_signal': normalized_audio,
                    'transcription': transcript
                })

                k+=1
                if k%5==0:
                    print(f"Iteratia {k}")

                if len(processed_data) >=batch:
                    save_to_h5(processed_data)
                    processed_data.clear()
        except Exception as e:
            print(f"Eroare la procesarea fisierului {audio_file}: {e}")

if processed_data:
    save_to_h5(processed_data)

print(f"Total fișiere procesate: {k}")