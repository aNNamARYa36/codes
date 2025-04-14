import os
import glob
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset

characters = "abcdefghijklmnopqrstuvwxyz '"
char_list = list(characters) 

char_to_index = {}  
for i, char in enumerate(char_list, start=1):  
    char_to_index[char] = i

index_to_char = {}  
for i, char in enumerate(char_list, start=1):
     index_to_char[i] = char

num_classes = len(char_list) + 1  

def transcript_to_int(text):
    text_lower = text.lower()
    indices = []
    for c in text_lower:
        index = char_to_index.get(c, 0)  
        indices.append(index)
    return indices

class SpeechDataset(Dataset):
    def __init__(self, folder):
        self.files = glob.glob(os.path.join(folder, "*.h5"))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        with h5py.File(self.files[idx], 'r') as hf:
            mel_list = []
            mel_lengths = []
            for key in hf['mel_spectrogram']:
                mel = hf['mel_spectrogram'][key][()]  #(128, time_steps)
                mel_tensor = torch.tensor(mel.T, dtype=torch.float32)  #(time_steps, 128)
                mel_list.append(mel_tensor)
                mel_lengths.append(mel_tensor.shape[0])
            
            transcriptions = []
            for key in hf['transcriptions']:
                text = hf['transcriptions'][key][()]
                text = text.decode('utf-8')
                transcriptions.append(text)
            labels_list = [transcript_to_int(text) for text in transcriptions]
            label_lengths = [len(label) for label in labels_list]
        
        return mel_list, labels_list, mel_lengths, label_lengths

def collate_fn(batch):
    all_mels = []
    all_labels = []
    all_mel_lengths = []
    all_label_lengths = []
    
    for mel_list, labels_list, mel_lengths, label_lengths in batch:
        for mel_tensor, label, mel_len, lab_len in zip(mel_list, labels_list, mel_lengths, label_lengths):
            all_mels.append(mel_tensor)
            all_labels.append(torch.tensor(label, dtype=torch.int32))
            all_mel_lengths.append(mel_len)
            all_label_lengths.append(lab_len)
    
    max_time = max(mel.shape[0] for mel in all_mels)
    padded_mels = []
    for mel in all_mels:
        pad_tensor = torch.zeros(max_time, mel.shape[1], dtype=mel.dtype)
        pad_tensor[:mel.shape[0], :] = mel
        padded_mels.append(pad_tensor)
    mel_batch = torch.stack(padded_mels)
    
    max_label = max(label.shape[0] for label in all_labels)
    padded_labels = []
    for label in all_labels:
        pad_label = torch.full((max_label,), -1, dtype=torch.int32)
        pad_label[:label.shape[0]] = label
        padded_labels.append(pad_label)
    label_batch = torch.stack(padded_labels)
    
    mel_lengths_tensor = torch.tensor(all_mel_lengths, dtype=torch.int32)
    label_lengths_tensor = torch.tensor(all_label_lengths, dtype=torch.int32)
    
    return mel_batch, label_batch, mel_lengths_tensor, label_lengths_tensor

class SpeechToTextModel(nn.Module):
    def __init__(self, input_dim=128, num_classes=num_classes):
        super(SpeechToTextModel, self).__init__()
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )
        
        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )
        
        self.cnn_block3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        
        self.cnn_block4 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        
        self.cnn_block5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )

        self.lstm1 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True, dropout=0.25)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True, dropout=0.25)
        self.lstm3 = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True, dropout=0.25)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.flatten = nn.Flatten()
        self.fc_dense = nn.Linear(256, 128)
        self.relu=nn.ReLU()
        self.out_layer = nn.Linear(128, num_classes)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() == 3 and x.size(1) != 128:
            x = x.transpose(1, 2)
    
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)
        x = self.cnn_block4(x)
        x = self.cnn_block5(x)   # După CNN: x are forma (batch, 128, T_reduced)
         
        x = x.transpose(1, 2)    #(batch, T_reduced, 128)
        
        x, _ = self.lstm1(x)   
        x, _ = self.lstm2(x)   
        x, _ = self.lstm3(x)   
        
        x = x.transpose(1, 2)   #(batch, 256, T_reduced)
        x = self.upsample(x)    
        x = x.transpose(1, 2)   #(batch, T_upsampled, 256)
        
        x = self.flatten(x)     #(batch, T_upsampled * 256)
        x = self.fc_dense(x)    #(batch, 128)
        x = self.relu(x)
        x = self.out_layer(x)   #(batch, num_classes)
        x = torch.nn.functional.log_softmax(x, dim=-1)
        return x
    
    def input_lengths(self, mel_lengths):
        pool1 = ((mel_lengths - 2) // 2) + 1
        pool2 = ((pool1 - 2) // 2) + 1
        out_length = pool2 * 2                 
        return out_length
        
def greedy_decoder(output, index_to_char, blank_index=0):
    arg_maxes = torch.argmax(output, dim=2).transpose(0, 1)  #(batch, time_steps)
    decoded = []
    for seq in arg_maxes:
        prev = blank_index
        output_seq = []
        for idx in seq:
            idx = idx.item()
            if idx != blank_index and idx != prev:
                output_seq.append(index_to_char.get(idx, ''))
            prev = idx
        decoded.append(''.join(output_seq))
    return decoded