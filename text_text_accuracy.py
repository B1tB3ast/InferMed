import os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import List, Tuple
import sys
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.transforms import InterpolationMode
from train import load_clip, preprocess_text
from zero_shot import make, make_true_labels, run_single_prediction

# Filepath for dataset
txt_filepath: str = 'train_data/mimic_impressions.csv'
cxr_filepath: str = 'train_data/cxr.h5'

# Model paths for different checkpoints
model_paths = [
    'checkpoints10gb50e/pt-imp/checkpoint_2000.pt',
    'checkpoints10gb50e/pt-imp/checkpoint_4000.pt',
    'checkpoints10gb50e/pt-imp/checkpoint_6000.pt',
    'checkpoints10gb50e/pt-imp/checkpoint_8000.pt',
    'checkpoints10gb50e/pt-imp/checkpoint_10000.pt',
    # 'checkpoints10gb50e/pt-imp/checkpoint_12000.pt',
    # 'checkpoints10gb50e/pt-imp/checkpoint_14000.pt',
    # 'checkpoints10gb50e/pt-imp/checkpoint_16000.pt',
    # 'checkpoints10gb50e/pt-imp/checkpoint_18000.pt',
]

class CXRDataset(Dataset):
    """Represents a CXR dataset with HDF5 images and CSV text reports."""
    def __init__(self, img_path, txt_path, column='impression', transform=None):
        super().__init__()
        self.h5_file = h5py.File(img_path, 'r')
        self.img_dset = self.h5_file['cxr']
        
        df = pd.read_csv(txt_path)
        self.filenames = df["filename"]
        self.txt_dset = df[column]

        self.transform = transform

    def __len__(self):
        return len(self.txt_dset)
    
    def __getitem__(self, idx):
        img = self.img_dset[idx]
        img = np.expand_dims(img, axis=0)  # Convert to (1, 320, 320)
        img = np.repeat(img, 3, axis=0)  # Convert to (3, 320, 320)
        
        txt = self.txt_dset[idx] if pd.notna(self.txt_dset[idx]) else " "
        study_id = self.filenames[idx]

        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        
        return {"img": img, "txt": txt, "id": study_id}

def load_data(cxr_filepath, txt_filepath, batch_size=16, column='impression', pretrained=False):
    """Loads CXR dataset and returns DataLoader."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944))
    ])

    if pretrained:
        transform.transforms.insert(0, transforms.Resize(224, interpolation=InterpolationMode.BICUBIC))
    
    torch_dset = CXRDataset(img_path=cxr_filepath, txt_path=txt_filepath, column=column, transform=transform)
    data_loader = data.DataLoader(torch_dset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return data_loader, device

# Load dataset and device
data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=16, pretrained=True)

# Load phenotype labels for evaluation
phenotype_label = pd.read_csv('/home/adimundada/Downloads/phenotype_labels.csv', dtype={'study_id': str})
phenotype_label['study_id'] = phenotype_label['study_id'].astype(str)
phenotype_label_dict = dict(zip(phenotype_label['study_id'], phenotype_label['phenotype_labels']))

# Store accuracy results for each model
text_to_text_accuracies = {}

for model_path in model_paths:
    print(f"Evaluating model: {model_path}")
    
    model = load_clip(model_path=model_path, pretrained=True, context_length=77)
    text_to_text_accuracy = []

    for batch in data_loader:
        images, texts, study_ids = batch['img'], batch['txt'], batch['id']
        
        transformed_ids = [study_id[1:-4] + ".0" for study_id in study_ids]

        images, texts = images.to(device), preprocess_text(texts, model).to(device)
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        count = 0
        for i in range(images.shape[0]):
            max_val = logits_per_image[i].max().item()
            index_arr = [transformed_ids[j] for j in range(images.shape[0]) if logits_per_image[i][j].item() == max_val]
            phenotypes = [phenotype_label_dict.get(index, "Unknown") for index in index_arr]
            actual_phenotype = phenotype_label_dict.get(transformed_ids[i], "Unknown")

            if actual_phenotype in phenotypes:
                count += 1

        text_to_text_accuracy.append(count / images.shape[0])

    avg_accuracy = np.mean(text_to_text_accuracy)
    text_to_text_accuracies[model_path] = avg_accuracy

    print(f"Average Text-to-Text Accuracy for {model_path}: {avg_accuracy}")

print("\nFinal Accuracy Results:")
for model, acc in text_to_text_accuracies.items():
    print(f"{model}: {acc}")
