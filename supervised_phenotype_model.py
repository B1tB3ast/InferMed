import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import csv
# Define the Fusion Model
class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 14)  # Output layer (14 classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, text_embedding, image_embedding):
        x = torch.cat((text_embedding, image_embedding), dim=1)  # Fusion of embeddings
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Logits output (no activation here)
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

def train_fusion_model(num_epochs=10, learning_rate=0.001, embeddings_path='embeddings.pth', save_path='checkpoints_supervised_model/'):
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
    csv_file = os.path.join(save_path, 'training_log_supervised.csv')
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
    train_fusion_model(num_epochs=50, learning_rate=0.001, embeddings_path=embeddings_path)
