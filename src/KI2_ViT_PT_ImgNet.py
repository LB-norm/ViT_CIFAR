from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import ToPILImage
from KI2_utils import get_cifar10_for_pretrained_VIT, train, validate, progress_callback
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import optuna
import pandas as pd
import numpy as np
import time
import os

current_dir = os.getcwd()
PT_ImgNet_ViT_path = os.path.join(current_dir, "models", "PT_ImgNet_ViT")
PT_ImgNet_ViT_HP_path = os.path.join(PT_ImgNet_ViT_path, "HPT.xlsx")
PT_ImgNet_ViT_save_path = os.path.join(PT_ImgNet_ViT_path, "PT_ImgNet.pth")

# 1. Load the pretrained ViT model and feature extractor
model_name = "google/vit-large-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=10)  # Adjust the number of labels
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(0)}")

def finetuning():
    #Second Phase
    learning_rate = 5e-6
    weight_decay = 7e-4     
    batch_size = 16 #Max für meinen GPU Speicher
    
    trainloader10, testloader10 = get_cifar10_for_pretrained_VIT(feature_extractor, batch_size)  #Load in Data
    
    model_name = "google/vit-large-patch16-224-in21k"
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=10)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    
    results_list = []
    best_acc = 0
    # Training loop
    for epoch in range(50):
        start_time = time.time()
        train_loss = train(model, trainloader10, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, testloader10, criterion, device)
        
        epoch_time = time.time() - start_time
        
        epoch_data =  {
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Val Accuracy": val_accuracy,
            "Time (seconds)": epoch_time
        }
        
        print(epoch_data)
        results_list.append(epoch_data)
            
        scheduler.step()
        
        if val_accuracy > best_acc:
            torch.save(model.state_dict(), PT_ImgNet_ViT_save_path)
            best_acc = val_accuracy
        
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(PT_ImgNet_ViT_path, "Large_model_results_50.xlsx")
    results_df.to_excel(results_path, index=False)

finetuning()


