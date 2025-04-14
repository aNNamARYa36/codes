import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from project_utils import SpeechDataset, collate_fn, SpeechToTextModel, greedy_decoder, index_to_char
from jiwer import wer

train_folder = "calea catre folder"  
val_folder   = "calea catre folder"
batch_size = 16

train_dataset = SpeechDataset(train_folder)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

val_dataset = SpeechDataset(val_folder)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeechToTextModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

def train_model(model, train_loader, val_loader, epochs=50):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for mel_specs, labels, mel_lengths, label_lengths in train_loader:
            mel_specs, labels = mel_specs.to(device), labels.to(device)
            outputs = model(mel_specs)  #(batch, time_steps, num_classes)
            outputs = outputs.permute(1, 0, 2)  # CTC Loss asteapta (time_steps, batch, num_classes)
            in_lengths=model.input_lengths(mel_lengths)
            loss = ctc_loss(outputs, labels, in_lengths, label_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoca {epoch+1}/{epochs} — Loss Antrenare: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        wer_scores = []
        with torch.no_grad():
            for mel_specs, labels, mel_lengths, label_lengths in val_loader:
                mel_specs, labels = mel_specs.to(device), labels.to(device)
                outputs = model(mel_specs)
                outputs = outputs.permute(1, 0, 2)
                in_lengths=model.input_lengths(mel_lengths)
                loss = ctc_loss(outputs, labels, in_lengths, label_lengths)
                total_val_loss += loss.item()
              
                decoded_preds = greedy_decoder(outputs, index_to_char=index_to_char, blank_index=0)
                targets = []
                for i, label in enumerate(labels):
                    label_seq = label[:label_lengths[i]]
                    text = ''.join([index_to_char.get(idx.item(), '') for idx in label_seq])
                    targets.append(text)
                for pred, target in zip(decoded_preds, targets):
                    wer_scores.append(wer(target, pred))
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_wer = sum(wer_scores) / len(wer_scores)
        print(f"Epoca {epoch+1}/{epochs} — Loss Validare: {avg_val_loss:.4f} — WER Validare: {avg_wer:.4f}")
    
    torch.save(model.state_dict(), "calea catre folder/modell.pth")
    print("Modelul a fost salvat.")

if __name__ == "__main__":
    train_model(model, train_loader, val_loader, epochs=50)
