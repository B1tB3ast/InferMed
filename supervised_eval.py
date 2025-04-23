from sklearn.metrics import f1_score
import torch.nn as nn
import torch
import numpy as np
import supervised_phenotype_model  # Import your training script

def evaluate_fusion_model(model, val_loader, device, threshold=0.5):
    """
    Evaluate the Fusion Model on the validation set.
    
    Args:
        model: Trained FusionModel instance.
        val_loader: DataLoader for validation data.
        device: CPU or GPU.
        threshold: Decision threshold for classification.
    
    Returns:
        Average loss, F1-score, and per-class accuracy.
    """
    model.eval()  # Set to evaluation mode
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():  # Disable gradient calculations
          # Initialize class sample counts
        
        for img_emb, txt_emb, lbl in val_loader:
            img_emb, txt_emb, lbl = img_emb.to(device).float(), txt_emb.to(device).float(), lbl.to(device).float()

            outputs = model(txt_emb, img_emb)
            loss = criterion(outputs, lbl)
            total_loss += loss.item()

            # Convert logits to probabilities using sigmoid
            probs = torch.sigmoid(outputs)

            # Apply threshold (0.5) to get binary predictions
            preds = (probs > threshold).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(lbl.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute F1-score (micro-average for multi-label classification)
    f1 = f1_score(all_labels, all_preds, average='micro')

    total_samples_per_class = np.sum(all_labels, axis=0)
    # Compute per-class accuracy
    per_class_accuracy = (all_preds == all_labels).sum(axis=0) / all_labels.shape[0]

    # Compute weighted overall accuracy
    weighted_accuracy = np.sum(per_class_accuracy * total_samples_per_class) / np.sum(total_samples_per_class)

    avg_loss = total_loss / len(val_loader)

    print(f"Validation Loss: {avg_loss:.4f}, F1-score: {f1:.4f}")
    print(f"Weighted Overall Accuracy: {weighted_accuracy:.4f}")
    print("Per-Class Accuracy:")
    for i, acc in enumerate(per_class_accuracy):
        print(f"Class {i}: {acc:.4f}")

    return avg_loss, f1, per_class_accuracy, weighted_accuracy



if __name__ == "__main__":
    embeddings_val_path = 'embeddings_test_with_labels.pth'

    # Load validation data
    val_loader, device = supervised_phenotype_model.load_data(embeddings_val_path)

    # Load the trained model
    model = supervised_phenotype_model.FusionModel().to(device)
    model.load_state_dict(torch.load("checkpoints_supervised_model/model_epoch_40.pth", map_location=device))  # Load last checkpoint

    # Evaluate the model
    evaluate_fusion_model(model, val_loader, device)
