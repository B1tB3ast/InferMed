import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
import os

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 14)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, text_embedding, image_embedding):
        x = torch.cat((text_embedding, image_embedding), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def load_data(embeddings_path, batch_size=4, labeled_fraction=1/3):
    data = torch.load(embeddings_path)
    image_embeddings = data['image_embeddings']
    text_embeddings = data['text_embeddings']
    labels = data['labels']

    total_samples = len(labels)
    indices = list(range(total_samples))

    try:
        # Try stratified split
        labeled_indices, unlabeled_indices = train_test_split(
            indices, train_size=labeled_fraction, stratify=labels, random_state=42
        )
    except ValueError as e:
        print("⚠️ Stratified split failed, falling back to random split.")
        labeled_indices, unlabeled_indices = train_test_split(
            indices, train_size=labeled_fraction, random_state=42
        )

    labeled_dataset = TensorDataset(
        image_embeddings[labeled_indices],
        text_embeddings[labeled_indices],
        labels[labeled_indices]
    )

    unlabeled_dataset = TensorDataset(
        image_embeddings[unlabeled_indices],
        text_embeddings[unlabeled_indices]
    )

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return labeled_loader, unlabeled_loader, device

def train_on_labeled(model, loader, device, num_epochs=50, save_path='checkpoints_labeled/'):
    os.makedirs(save_path, exist_ok=True)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for img_emb, txt_emb, lbl in loader:
            img_emb, txt_emb, lbl = img_emb.to(device).float(), txt_emb.to(device).float(), lbl.to(device)
            optimizer.zero_grad()
            outputs = model(txt_emb, img_emb)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Labeled] Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(loader):.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))
            print(f"Checkpoint saved: model_epoch_{epoch+1}.pth")

def generate_pseudo_labels(model, loader, device, threshold=1):
    model.eval()
    pseudo_data = []

    with torch.no_grad():
        for img_emb, txt_emb in loader:
            img_emb, txt_emb = img_emb.to(device).float(), txt_emb.to(device).float()
            logits = model(txt_emb, img_emb)
            probs = torch.sigmoid(logits)

            confident_mask = (probs > threshold).float()
            confident_samples = (confident_mask.sum(dim=1) == probs.size(1))

            pseudo_imgs = img_emb[confident_samples]
            pseudo_txts = txt_emb[confident_samples]
            pseudo_labels = probs[confident_samples].round()

            for i in range(len(pseudo_imgs)):
                pseudo_data.append((pseudo_imgs[i], pseudo_txts[i], pseudo_labels[i]))

    print(f"Generated {len(pseudo_data)} pseudo-labeled samples with confidence > {threshold}")
    return pseudo_data

def retrain_on_combined(model, labeled_loader, pseudo_data, device, num_epochs=50, save_path='checkpoints_retrain/'):
    os.makedirs(save_path, exist_ok=True)
    model.train()

    if pseudo_data:
        imgs, txts, lbls = zip(*pseudo_data)
        pseudo_dataset = TensorDataset(torch.stack(imgs), torch.stack(txts), torch.stack(lbls))
        combined_dataset = ConcatDataset([labeled_loader.dataset, pseudo_dataset])
    else:
        combined_dataset = labeled_loader.dataset

    combined_loader = DataLoader(combined_dataset, batch_size=4, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for img_emb, txt_emb, lbl in combined_loader:
            img_emb, txt_emb, lbl = img_emb.to(device).float(), txt_emb.to(device).float(), lbl.to(device)
            optimizer.zero_grad()
            outputs = model(txt_emb, img_emb)
            loss = criterion(outputs, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Retrain] Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(combined_loader):.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))
            print(f"Checkpoint saved: model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    EMBEDDINGS_PATH = 'embeddings_train_with_labels.pth'
    CONFIDENCE_THRESHOLD = 0.5

    labeled_loader, unlabeled_loader, device = load_data(EMBEDDINGS_PATH)

    model = FusionModel().to(device)

    print("🔹 Step 1: Training on labeled data...")
    train_on_labeled(model, labeled_loader, device, num_epochs=50)

    print("🔹 Step 2: Generating pseudo-labels...")
    pseudo_data = generate_pseudo_labels(model, unlabeled_loader, device, threshold=CONFIDENCE_THRESHOLD)

    print("🔹 Step 3: Retraining on labeled + pseudo-labeled data...")
    retrain_on_combined(model, labeled_loader, pseudo_data, device, num_epochs=50)

    torch.save(model.state_dict(), 'checkpoints_retrain/final_model_semi_supervised.pth')
    print("✅ Training complete. Final model saved.")
