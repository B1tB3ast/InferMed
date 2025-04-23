import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from train import load_clip, preprocess_text, load_data
import sys
import torch

# from eval import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_single_prediction

 # filepath of chest x-ray images (.h5)

model_dir: str = '../checkpoints/chexzero_weights' # where pretrained models are saved (.pt) 
predictions_dir: Path = Path('../predictions') # where to save predictions
cache_dir: str = predictions_dir / "cached" # where to cache ensembled predictions
cxr_pair_template: Tuple[str] = ("{}", "no {}")
context_length: int = 77
save_name: str = None
# ------- LABELS ------  #
# Define labels to query each image | will return a prediction for each label
cxr_labels: List[str] = ['Atelectasis','Cardiomegaly', 
                                      'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                      'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
                                      'Pneumothorax', 'Support Devices']

predictions = []
model_paths = ['checkpoints_train/pt-imp/checkpoint_18000.pt',]
txt_filepath: str = 'test_data/mimic_impressions.csv'
cxr_filepath: str = 'test_data/cxr.h5'

data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=16, pretrained=True, column="impression")
model = load_clip(model_path=model_paths[0], pretrained=True, context_length=77)
batch_size=16

al=[]
for data in data_loader:
    images=data['img']
    texts=data['txt']
    
    texts=preprocess_text(texts,model)
    images, texts = images.to(device), texts.to(device)
    image_features = model.encode_image(images)
    text_features = model.encode_text(texts)

        # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()
    # print(logits_per_image.shape)
    # print(logits_per_image)

    # count=0
    # y_pred=torch.argmax(logits_per_image, dim=1)
    # # print(y_pred)
    
    # for i in range(batch_size):
    #     if(y_pred[i]==i):
    #         count+=1
    # print("Alignment accuracy= ",count/batch_size)
    batch_size = images.shape[0]  # Handle last batch case
    y_pred = torch.argmax(logits_per_image, dim=1)
    count=0
    for i in range(images.shape[0]):
        max_val=0
        flag=0
        for j in range(images.shape[0]):
            if(logits_per_image[i][j]>max_val):
                max_val=logits_per_image[i][j]
                
        if(logits_per_image[i][i]==max_val):
            flag=1
        
        if(flag==1):
            count+=1
    correct = torch.arange(batch_size).to(device)
    # matches = (y_pred == correct).sum().item()
    alignment_accuracy = count / batch_size
    al.append(alignment_accuracy)
    # print(f"Batch Size: {batch_size}, Alignment Accuracy: {alignment_accuracy:.4f}")
print("Average accuracy: ",np.mean(al))