import os
import pprint
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize
import csv
import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

from train import train_main, load_data, load_clip, preprocess_text
from zero_shot import run_cxr_zero_shot, run_zero_shot

align_log="align_logging.csv"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='train_data/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--txt_filepath', type=str, default='train_data/mimic_impressions.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default="checkpoints_train/", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="pt-imp")
    args = parser.parse_args()
    return args

def model_pipeline(config, verbose=0): 
    # make the model, data, and optimization problem
    model, data_loader, device, criterion, optimizer = make(config)

    # and use them to train the model
    train(model, data_loader, device, criterion, optimizer, config)

    # save model
    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
    save(model, model_path)

    if verbose: 
        print(model)
    return model

def make(config): 
    pretrained = not config.random_init
    data_loader, device = load_data(config.cxr_filepath, config.txt_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression")
    model = load_clip(model_path=None, pretrained=pretrained, context_length=config.context_length)
    model.to(device)
    print('Model on Device.')

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    if config.optimizer == "adam": 
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    return model, data_loader, device, criterion, optimizer

def train(model, loader, device, criterion, optimizer, config): 
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(model_save_dir): 
        # Create a new folder if not exists
        os.makedirs(model_save_dir)
    log_file = "logging_info.csv"
    

# Overwrite the file to reset it at the start of each run
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Checkpoints", "Running Loss per report_freq"])  # Write header again

    with open(align_log, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Batch", "Alignment_accuracy"])  # Write header again

    # Run training
    total_batches = len(loader) * config.epochs
    loss_history = []
    example_ct = 0  # number of examples seen
    batch_ct = 0
    report_freq = config.log_interval
    highest_val_auc = 0 # save highest mean auc
    
    for epoch in range(config.epochs):
        running_loss = 0.0 # running loss over batch
        for data in tqdm(loader):
            # get the images
            images = data['img']

            texts = data['txt']
            texts = preprocess_text(texts, model) 
            # print(type(texts))
            
            # perform step for a single batch
            loss = train_batch(images, texts, model, device, criterion, optimizer,batch_ct)
            example_ct +=  len(images)
            batch_ct += 1
            running_loss += loss.item()

            # Report metrics every `report_freq` batch
            if (batch_ct % report_freq) == 0:
                train_log(running_loss / report_freq, example_ct, epoch)
                with open(log_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([example_ct, running_loss/report_freq])
                running_loss = 0.0
                
            if (batch_ct % config.save_interval) == 0: 
                model_path = os.path.join(model_save_dir, "checkpoint_{batch_ct}.pt".format(
                    batch_ct=str(batch_ct), 
                ))
                print("Saved checkpoint to: ", model_path)
                save(model, model_path)
        

        # print(f"Epoch {epoch+1}/{config.epochs}, Running Loss: {running_loss:.4f}")


def alignment_accuracy(logits_per_image,batch_size):
    count=0
    for i in range(batch_size):
        max_val=0
        flag=0
        for j in range(batch_size):
            if(logits_per_image[i][j]>max_val):
                max_val=logits_per_image[i][j]
                
        if(logits_per_image[i][i]==max_val):
            flag=1
        
        if(flag==1):
            count+=1

    alignment_accuracy = count / batch_size
    # print(f"Batch Size: {batch_size}, Alignment Accuracy: {alignment_accuracy:.4f}")
    return(alignment_accuracy)


def train_batch(images, texts, model, device, criterion, optimizer,batch_ct):
    images, texts = images.to(device), texts.to(device)
    
    # Forward pass ➡
    logits_per_image, logits_per_text = model(images, texts)
    align_accuracy=alignment_accuracy(logits_per_image,images.shape[0])
    with open(align_log, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([batch_ct, align_accuracy])
    # Create labels
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(device)
    
    # Compute loss
    loss_img = criterion(logits_per_image, labels)
    loss_txt = criterion(logits_per_text, labels)
    loss = (loss_img + loss_txt)/2 # avg. img and txt loss

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()
    
    # Step with optimizer
    optimizer.step()
        
    return loss

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
def save(model, path): 
    torch.save(model.state_dict(), path)
    
if __name__ == "__main__":
    args = parse_args()
    model = model_pipeline(args)
    

