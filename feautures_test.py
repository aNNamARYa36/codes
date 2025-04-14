import h5py
import librosa
import numpy as np
import os

h5_folder = "D:\\Anul 4\\LICENTA stuff\\date_procesate\\date_proc_test_h5_new"
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

# h5_folder = "D:\\Anul 4\\LICENTA stuff\\date_procesate\\date_proc_test_h5"
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

