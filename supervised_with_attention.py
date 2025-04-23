import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import csv
import torch.nn.functional as F
# Define the Fusion Model

# class FusionModelWithScalarAttention(nn.Module):
#     def __init__(self):
#         super(FusionModelWithScalarAttention, self).__init__()
        
#         self.text_proj = nn.Linear(512, 512)
#         self.image_proj = nn.Linear(512, 512)

#         self.attn_weights = nn.Parameter(torch.randn(2))  # Learnable scalar weights

#         self.fc1 = nn.Linear(512, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 14)

#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, text_embedding, image_embedding):
#         text = self.text_proj(text_embedding)
#         image = self.image_proj(image_embedding)

#         # Scalar attention (softmax over 2 modalities)
#         attn = F.softmax(self.attn_weights, dim=0)  # [2]
#         fused = attn[0] * text + attn[1] * image    # Weighted sum (B, 512)

#         x = self.relu(self.fc1(fused))
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc3(x))
#         x = self.fc4(x)

#         return x

class FusionModel(nn.Module):
    def __init__(self, text_dim=512, image_dim=512, common_dim=256, num_heads=8, dropout=0.3, num_classes=14):
        super(FusionModel, self).__init__()
        self.text_proj = nn.Linear(text_dim, common_dim)
        self.image_proj = nn.Linear(image_dim, common_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(common_dim)
        self.norm2 = nn.LayerNorm(common_dim)
        self.fc1 = nn.Linear(common_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_embedding, image_embedding):
        # Project embeddings to common dimension
        text_embedding = self.text_proj(text_embedding).unsqueeze(1)  # Shape: (B, 1, common_dim)
        image_embedding = self.image_proj(image_embedding).unsqueeze(1)  # Shape: (B, 1, common_dim)

        # Image attends to text: Query=image_embedding, Key=Value=text_embedding
        attn_output, _ = self.cross_attention(query=image_embedding, key=text_embedding, value=text_embedding)
        attn_output = self.norm1(attn_output + image_embedding)  # Residual connection and normalization

        # Pass through feedforward network
        x = self.relu(self.fc1(attn_output.squeeze(1)))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output logits for each class

        return x
    
def load_data(embeddings_path='embeddings.pth', batch_size=4):
    """
    Load saved embeddings and create a DataLoader.
    """
    # Load the saved embeddings
    data = torch.load(embeddings_path)
    
    image_embeddings = data['image_embeddings']
    text_embeddings = data['text_embeddings']
    labels = data['labels']

    # Move to device (if necessary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to dataset & dataloader
    dataset = TensorDataset(image_embeddings, text_embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader, device

def train_fusion_model(num_epochs=10, learning_rate=0.001, embeddings_path='embeddings.pth', save_path='checkpoints_supervised_model_attention/'):
    """
    Train the Fusion Model using precomputed embeddings with model checkpointing and CSV logging.
    """
    # Ensure checkpoint directory exists
    os.makedirs(save_path, exist_ok=True)

    # Load data
    dataloader, device = load_data(embeddings_path)

    # Initialize model, loss, and optimizer
    model = FusionModel().to(device)
    criterion = nn.BCEWithLogitsLoss()  # For multilabel classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create or open CSV file for logging
    csv_file = os.path.join(save_path, 'training_log_supervised_attention.csv')
    write_header = not os.path.exists(csv_file)  # Check if file exists to avoid duplicate headers

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['Epoch', 'Loss'])  # Write header only if file is new

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for img_emb, txt_emb, lbl in dataloader:
                img_emb, txt_emb, lbl = img_emb.to(device).float(), txt_emb.to(device).float(), lbl.to(device)
                optimizer.zero_grad()
                outputs = model(txt_emb, img_emb)
                loss = criterion(outputs, lbl)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

            # Write epoch loss to CSV
            writer.writerow([epoch + 1, avg_loss])
            file.flush()  # Ensure data is written to file immediately

            # Save checkpoint every 2 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_path, f'model_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

    print("Training complete!")
if __name__ == "__main__":
    embeddings_path='embeddings_train_with_labels.pth'
    train_fusion_model(num_epochs=60, learning_rate=0.001, embeddings_path=embeddings_path)
