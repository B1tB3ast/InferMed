import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import List, Tuple
import sys
import torch
torch.cuda.empty_cache()
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.utils.data as data
from torchvision.transforms import InterpolationMode
from train import load_clip, preprocess_text

txt_filepath: str = 'test_data/mimic_impressions.csv'
cxr_filepath: str = 'test_data/cxr.h5'
pheno_path: str='phenotype_labels.csv'
model_path: str='checkpoints_train/pt-imp/checkpoint_4000.pt'


class CXRDataset(Dataset):
    """Represents a CXR dataset with HDF5 images, CSV text reports, and phenotype labels."""
    PHENOTYPES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                  'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                  'Pneumothorax', 'Support Devices']
    
    def __init__(self, img_path, txt_path, pheno_path, column='impression', transform=None):
        super().__init__()
        self.h5_file = h5py.File(img_path, 'r')
        self.img_dset = self.h5_file['cxr']
        
        df = pd.read_csv(txt_path)
        self.filenames = df["filename"]
        self.txt_dset = df[column]
        
        pheno_df = pd.read_csv(pheno_path, dtype={'study_id': str})
        pheno_df['study_id'] = pheno_df['study_id'].astype(str)
        self.phenotype_dict = dict(zip(pheno_df['study_id'], pheno_df['phenotype_labels']))
        
        self.transform = transform
    
    def one_hot_encode(self, labels):
        label_vector = np.zeros(len(self.PHENOTYPES), dtype=np.float32)
        if isinstance(labels, str):

            
            for label in labels.split(','):
                label = label.strip()

                
                if label in self.PHENOTYPES:
                    label_vector[self.PHENOTYPES.index(label)] = 1.0
        return label_vector
    
    def __len__(self):
        return len(self.txt_dset)
    
    def __getitem__(self, idx):
        img = self.img_dset[idx]
        img = np.expand_dims(img, axis=0)  # Convert to (1, 320, 320)
        img = np.repeat(img, 3, axis=0)  # Convert to (3, 320, 320)
        
        txt = self.txt_dset[idx] if pd.notna(self.txt_dset[idx]) else " "
        study_id = self.filenames[idx]
        transformed_id = study_id[1:-4] + ".0"
        phenotype_labels = self.phenotype_dict.get(transformed_id, "")

        phenotype_vector = self.one_hot_encode(phenotype_labels)
        
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        
        return {"img": img, "txt": txt, "label": torch.tensor(phenotype_vector)}
    
def load_data(cxr_filepath, txt_filepath, batch_size=16, column='impression', pretrained=False):
    """Loads CXR dataset and returns DataLoader."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944))
    ])

    if pretrained:
        transform.transforms.insert(0, transforms.Resize(224, interpolation=InterpolationMode.BICUBIC))
    
    torch_dset = CXRDataset(img_path=cxr_filepath, txt_path=txt_filepath,pheno_path=pheno_path, column=column, transform=transform)
    data_loader = data.DataLoader(torch_dset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return data_loader, device
data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=4, pretrained=True)

model = load_clip(model_path=model_path, pretrained=True, context_length=77)
image_embeddings_list = []
text_embeddings_list = []
labels_list = []
with torch.no_grad():
    for batch in data_loader:
        images, texts, labels = batch['img'], batch['txt'], batch['label']
        

        images, texts = images.to(device), preprocess_text(texts, model).to(device)
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = image_features.detach().cpu()  # Free GPU memory
        text_features = text_features.detach().cpu()
        labels = labels.detach().cpu()

        image_embeddings_list.append(image_features)
        text_embeddings_list.append(text_features)
        labels_list.append(labels)
        torch.cuda.empty_cache()

image_embeddings_tensor = torch.cat(image_embeddings_list, dim=0)
text_embeddings_tensor = torch.cat(text_embeddings_list, dim=0)
labels_tensor = torch.cat(labels_list, dim=0)

torch.save({
    'image_embeddings': image_embeddings_tensor,
    'text_embeddings': text_embeddings_tensor,
    'labels': labels_tensor
}, 'embeddings_test_with_labels.pth')

print("Embeddings saved successfully!")