from sklearn.metrics import f1_score
import torch.nn as nn
import torch
import numpy as np
import supervised_phenotype_model  # Import your training + model module


def load_data_for_eval(embeddings_path, batch_size=4):
    """
    Loads full labeled data for evaluation (no train-test split).
    """
    data = torch.load(embeddings_path)
    image_embeddings = data['image_embeddings']
    text_embeddings = data['text_embeddings']
    labels = data['labels']

    dataset = torch.utils.data.TensorDataset(image_embeddings, text_embeddings, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return loader, device


def evaluate_fusion_model(model, val_loader, device, threshold=0.5):
    """
    Evaluate the Fusion Model on the validation set.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for img_emb, txt_emb, lbl in val_loader:
            img_emb, txt_emb, lbl = img_emb.to(device).float(), txt_emb.to(device).float(), lbl.to(device).float()

            outputs = model(txt_emb, img_emb)
            loss = criterion(outputs, lbl)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(lbl.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    f1 = f1_score(all_labels, all_preds, average='micro')

    # Per-class accuracy
    correct = (all_preds == all_labels)
    total_per_class = np.sum(all_labels, axis=0)
    per_class_accuracy = np.sum(correct, axis=0) / all_labels.shape[0]

    weighted_accuracy = np.sum(per_class_accuracy * total_per_class) / np.sum(total_per_class)
    avg_loss = total_loss / len(val_loader)

    print(f"\nValidation Loss: {avg_loss:.4f}")
    print(f"Micro F1-score: {f1:.4f}")
    print(f"Weighted Overall Accuracy: {weighted_accuracy:.4f}")
    print("Per-Class Accuracy:")
    for i, acc in enumerate(per_class_accuracy):
        print(f"Class {i}: {acc:.4f}")

    return avg_loss, f1, per_class_accuracy, weighted_accuracy


if __name__ == "__main__":
    embeddings_val_path = 'embeddings_train_with_labels.pth'

    val_loader, device = load_data_for_eval(embeddings_val_path)

    model = supervised_phenotype_model.FusionModel().to(device)
    model.load_state_dict(torch.load("checkpoints_retrain/final_model_semi_supervised.pth", map_location=device))

    evaluate_fusion_model(model, val_loader, device)
