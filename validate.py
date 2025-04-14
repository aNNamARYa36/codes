# validate.py

import os
import torch
from torch.utils.data import DataLoader
from project_utils import SpeechDataset, collate_fn, SpeechToTextModel, greedy_decoder, index_to_char
from jiwer import wer

# Setează calea către folderul de validare din Google Drive
val_folder = "/content/drive/MyDrive/date_proc_dev_h5"  # modifică calea după necesitate
batch_size = 16

# Crează dataset-ul și DataLoader-ul pentru validare
val_dataset = SpeechDataset(val_folder)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeechToTextModel().to(device)
# Încarcă modelul antrenat
model.load_state_dict(torch.load("modell.pth", map_location=device))
model.eval()
ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

def validate_model(model, val_loader):
    val_loss = 0
    wer_scores = []
    with torch.no_grad():
        for mel_specs, labels, input_lengths, label_lengths in val_loader:
            mel_specs, labels = mel_specs.to(device), labels.to(device)
            outputs = model(mel_specs)
            outputs = outputs.permute(1, 0, 2)
            loss = ctc_loss(outputs, labels, input_lengths, label_lengths)
            val_loss += loss.item()
            
            decoded_preds = greedy_decoder(outputs, index_to_char=index_to_char, blank_index=0)
            targets = []
            for i, label in enumerate(labels):
                label_seq = label[:label_lengths[i]]
                text = ''.join([index_to_char.get(idx.item(), '') for idx in label_seq])
                targets.append(text)
            for pred, target in zip(decoded_preds, targets):
                wer_scores.append(wer(target, pred))
    
    avg_val_loss = val_loss / len(val_loader)
    avg_wer = sum(wer_scores) / len(wer_scores)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation WER: {avg_wer:.4f}")

if __name__ == "__main__":
    # Asigură-te că ai montat Google Drive înainte de a rula acest script
    validate_model(model, val_loader)
