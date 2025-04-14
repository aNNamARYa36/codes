import torch
from torch.utils.data import DataLoader
from project_utils import SpeechDataset, collate_fn, SpeechToTextModel, greedy_decoder, index_to_char
from jiwer import wer

test_folder = "calea catre folder"  
batch_size = 16

test_dataset = SpeechDataset(test_folder)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpeechToTextModel().to(device)
model.load_state_dict(torch.load("modell.pth", map_location=device))
model.eval()

def evaluate_on_test(model, test_loader):
    wer_scores = []
    with torch.no_grad():
        for mel_specs, labels, _, label_lengths in test_loader:
            mel_specs, labels = mel_specs.to(device), labels.to(device)
            outputs = model(mel_specs)
            outputs = outputs.permute(1, 0, 2)
            decoded_preds = greedy_decoder(outputs, index_to_char=index_to_char, blank_index=0)
            targets = []
            for i, label in enumerate(labels):
                label_seq = label[:label_lengths[i]]
                text = ''.join([index_to_char.get(idx.item(), '') for idx in label_seq])
                targets.append(text)
            for pred, target in zip(decoded_preds, targets):
                wer_scores.append(wer(target, pred))
    avg_wer = sum(wer_scores) / len(wer_scores)
    print(f"\nFinal Test WER: {avg_wer:.4f}")

if __name__ == "__main__":
    evaluate_on_test(model, test_loader)
