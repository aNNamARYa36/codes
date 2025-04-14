import h5py
import librosa
import numpy as np
import os

h5_folder = "D:\\Anul 4\\LICENTA stuff\\date_procesate\\date_proc_dev_h5_new"
k=0

for filename in os.listdir(h5_folder):
    if filename.endswith(".h5"):  
        h5_filename = os.path.join(h5_folder, filename)

        with h5py.File(h5_filename, "a") as h5f:
            audio_signals = h5f["audio_signals"]

            # if "mfcc" not in h5f:
            #     mfcc_group = h5f.create_group("mfcc")
            # else:
            #     mfcc_group = h5f["mfcc"]

            # if "delta_mfcc" not in h5f:
            #     delta_group = h5f.create_group("delta_mfcc")
            # else:
            #     delta_group = h5f["delta_mfcc"]

            # if "delta_delta_mfcc" not in h5f:
            #     delta_delta_group = h5f.create_group("delta_delta_mfcc")
            # else:
            #     delta_delta_group = h5f["delta_delta_mfcc"]

            if "mel_spectrogram" not in h5f:
                mel_group = h5f.create_group("mel_spectrogram")
            else:
                mel_group = h5f["mel_spectrogram"]

            for id in audio_signals.keys():
                audio_signal = np.array(audio_signals[id])

                # mfcc = librosa.feature.mfcc(y=audio_signal, sr=16000, n_mfcc=13)
                # delta_mfcc = librosa.feature.delta(mfcc)
                # delta_delta_mfcc = librosa.feature.delta(mfcc, order=2)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_signal, sr=16000, n_mels=128)
                mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

                # if id in mfcc_group:
                #     del mfcc_group[id]
                # mfcc_group.create_dataset(id, data=mfcc, dtype=np.float32)

                # if id in delta_group:
                #     del delta_group[id]
                # delta_group.create_dataset(id, data=delta_mfcc, dtype=np.float32)

                # if id in delta_delta_group:
                #     del delta_delta_group[id]
                # delta_delta_group.create_dataset(id, data=delta_delta_mfcc, dtype=np.float32)

                if id in mel_group:
                    del mel_group[id]
                mel_group.create_dataset(id, data=mel_spectrogram_db, dtype=np.float32)

                k+=1
                if k%5==0:
                    print(f"iteratia {k}")

print("Procesare completă")



# import h5py
# import librosa
# import numpy as np
# import os

# h5_folder = "D:\\Anul 4\\LICENTA stuff\\date_procesate\\date_proc_dev_h5"
# k=0

# for filename in os.listdir(h5_folder):
#     if filename.endswith(".h5"):  
#         h5_filename = os.path.join(h5_folder, filename)

#         with h5py.File(h5_filename, "a") as h5f:
#             if "mfcc" in h5f:
#                 del h5f["mfcc"]

#             if "delta_mfcc" in h5f:
#                 del h5f["delta_mfcc"]
            
#             if "delta_delta_mfcc" in h5f:
#                 del h5f["delta_delta_mfcc"]

#             k+=1
#             if k%5==0:
#                 print(f"iteratia {k}")
# print("stergere completa!")



# Deschidem fișierul .h5
# with h5py.File("D:\\Anul 4\\LICENTA stuff\\date_procesate\\date_proc_dev_h5\\dev_proc_data_0.h5", "r") as f:
    
#     count = 0  # Contor pentru a limita output-ul

#     # Iterăm prin toate dataset-urile din grupul 'transcriptions'
#     for key in f['mel_spectrogram']:
#         mel = f['mel_spectrogram'][key][()]  # Citim dataset-ul
#         print(f"Dataset: {key}, format: {mel.shape}")  # Afișăm tipul fiecărei transcrieri

#         count += 1  # Incrementăm contorul
#         if count >= 5:  # Oprim după 5 dataset-uri
#             break


# with h5py.File("D:\\Anul 4\\LICENTA stuff\\date_procesate\\date_proc_dev_h5\\dev_proc_data_0.h5", 'r') as hf:
#     print("Keys in HDF5 file:", list(hf.keys()))
#     print("Structure of transcriptions:", hf['transcriptions'])

#     print("Keys in transcriptions group:", list(hf['transcriptions'].keys()))
#     for key in hf['transcriptions']:
#         print(f"Dataset '{key}' type: {type(hf['transcriptions'][key])}")



# import os
# import glob
# import h5py

# def get_max_mel_length(folders):
#     max_length = 0
#     for folder in folders:
#         # Căutăm toate fișierele .h5 din folder
#         h5_files = glob.glob(os.path.join(folder, '*.h5'))
#         for file in h5_files:
#             with h5py.File(file, 'r') as hf:
#                 # Verificăm dacă grupul 'mel_spectrogram' există
#                 if 'mel_spectrogram' in hf:
#                     for key in hf['mel_spectrogram']:
#                         mel = hf['mel_spectrogram'][key][()]  # forma este (128, time_steps)
#                         time_steps = mel.shape[1]  # time_steps este pe axa 1
#                         if time_steps > max_length:
#                             max_length = time_steps
#     return max_length

# # Exemplu de listă cu foldere
# folders = [
#     "D:\\Anul 4\\LICENTA stuff\\date_procesate\\date_proc_train100_h5",
#     "D:\\Anul 4\\LICENTA stuff\\date_procesate\\date_proc_dev_h5",
#     "D:\\Anul 4\\LICENTA stuff\\date_procesate\\date_proc_test_h5"
# ]

# max_mel_length = get_max_mel_length(folders)
# print("Lungimea maximă a mel spectrogramelor:", max_mel_length)

