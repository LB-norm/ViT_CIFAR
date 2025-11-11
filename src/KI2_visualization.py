import pandas as pd
import numpy as np
import time
import os
import torch
import torchvision
import optuna
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from KI2_utils import get_cifar10_data,get_cifar100_data, linear_warmup_decay, train, pretrain, validate, progress_callback
from KI2_VIT_from_Scratch import VisionTransformer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

current_dir = os.getcwd()
base_ViT_path = os.path.join(current_dir, "models", "base_ViT")
base_ViT_early_HP_path = os.path.join(base_ViT_path, "HPT_early_testing.xlsx")
base_ViT_intermediate_HP_path = os.path.join(base_ViT_path, "HPT_intermediate_testing.xlsx")

PT_C100_path = os.path.join(current_dir, "models", "PT_CIFAR100_ViT")
PT_C100_PT_HP_path = os.path.join(PT_C100_path, "HPT_intermediate_testing.xlsx")
PT_C100_PT_FT_path = os.path.join(PT_C100_path, "HPT_finetuning_intermediate_testing.xlsx")

large_ViT_path = os.path.join(current_dir, "models", "PT_ImgNet_ViT")

def show_transformation_pipeline():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ])
        
        
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    
    image, _ = testset[1]  # Get the 3rd image (index 2)
    
    # Define individual transformations
    transform_flip = transforms.RandomHorizontalFlip(p=1.0)  # Always flip for demonstration
    transform_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
    transform_rotate = transforms.RandomRotation(15)
    transform_grayscale = transforms.RandomGrayscale(p=1.0)  # Always grayscale for demonstration
    
    # Normalization parameters
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.247, 0.243, 0.261]
    
    # Apply transformations individually
    image_flip = transform_flip(image)
    image_jitter = transform_jitter(image)
    image_rotate = transform_rotate(image)
    image_grayscale = transform_grayscale(image)
    
    # Prepare to plot each transformed image
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    
    # Original image
    axs[0].imshow(image.permute(1, 2, 0))
    axs[0].set_title('Original')
    axs[0].axis('off')
    
    # Flipped image
    axs[1].imshow(image_flip.permute(1, 2, 0))
    axs[1].set_title('Flipped') 
    axs[1].axis('off')
    
    # Color jittered image
    axs[2].imshow(image_jitter.permute(1, 2, 0))
    axs[2].set_title('Color Jitter')
    axs[2].axis('off')
    
    # Rotated image
    axs[3].imshow(image_rotate.permute(1, 2, 0))
    axs[3].set_title('Rotated')
    axs[3].axis('off')
    
    # Grayscale image
    axs[4].imshow(image_grayscale.permute(1, 2, 0), cmap='gray')
    axs[4].set_title('Grayscale')
    axs[4].axis('off')
    
    plt.tight_layout()
    plt.show()

# show_transformation_pipeline()

def add_jitter(df, categorical_cols):
    small_jitter = 0.05
    df_new = df.copy()
    for col in df_new.columns:
        if col in categorical_cols:
            df_new[col] += np.random.uniform(-small_jitter, small_jitter, size=len(df_new))
    return df_new
    
Base_Early_HPT = pd.read_excel(base_ViT_early_HP_path)
categorical_cols = ["batch_size", "num_layers", "num_heads", "emb_size", "ff_hidden_size"]
Base_Early_HPT_jittered = add_jitter(Base_Early_HPT, categorical_cols)
Base_Early_HPT_jittered["validation_accuracy"] = Base_Early_HPT_jittered["validation_accuracy"].fillna(0)

Base_Int_HPT = pd.read_excel(base_ViT_intermediate_HP_path)
Base_Int_HPT_jittered = add_jitter(Base_Int_HPT, categorical_cols)
Base_Int_HPT_jittered["validation_accuracy"] = Base_Int_HPT_jittered["validation_accuracy"].fillna(0)

PT_Int_PT_df = pd.read_excel(PT_C100_PT_HP_path)
PT_categorical_cols = ["batch_size"]
PT_Int_PT_HPT_jittered = add_jitter(PT_Int_PT_df, PT_categorical_cols)
PT_Int_PT_HPT_jittered["validation_accuracy"] = PT_Int_PT_HPT_jittered["validation_accuracy"].fillna(0)



def visualize_HPT(df, df_jittered, categorical=False):
    if categorical == True:        
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=df_jittered['validation_accuracy'],  
                    colorscale='Cividis',  #Cividis
                    showscale=True
                ),
                dimensions=[
                    dict(label="batch_size", values=df_jittered["batch_size"], tickvals=sorted(df["batch_size"].unique())),
                    # dict(label="num_layers", values=df_jittered["num_layers"], tickvals=sorted(df["num_layers"].unique())),
                    # dict(label="num_heads", values=df_jittered["num_heads"], tickvals=sorted(df["num_heads"].unique())),
                    # dict(label="emb_size", values=df_jittered["emb_size"], tickvals=sorted(df["emb_size"].unique())),
                    # dict(label="ff_hidden_size", values=df_jittered["ff_hidden_size"], tickvals=sorted(df["ff_hidden_size"].unique())),
                    dict(label="dropout_rate", values=df["dropout_rate"], tickvals=[]),
                    dict(label="weight_decay", values=df["weight_decay"]),  
                    dict(label="learning_rate", values=df["learning_rate"]),
                    dict(label="validation_accuracy", values=df_jittered["validation_accuracy"])
                ]
            )
        )
            
        fig.update_layout(title='Pretraining Study 10 Epochs', font=dict(size=22))
        fig.show()
   
#     dimensions = [
#         dict(label="dropout_rate", values=df["dropout_rate"]),
#         dict(label="weight_decay", values=df["weight_decay"]),  
#         dict(label="learning_rate", values=df["learning_rate"])
#     ]
# visualize_HPT(Base_Early_HPT, Base_Early_HPT_jittered, categorical=True)
# visualize_HPT(Base_Int_HPT, Base_Int_HPT_jittered, categorical=True)
# visualize_HPT(PT_Int_PT_df, PT_Int_PT_HPT_jittered, categorical=True )

def plot_model_metrics(df):
    # Create a figure
    fig = go.Figure()

    # Add Train Loss as a histogram
    fig.add_trace(
        go.Bar(
            x=df['Epoch'],
            y=df['Train Loss'],
            name='Train Loss',
            marker=dict(color='blue'),
            opacity=0.6,
            yaxis='y1'
        )
    )

    # Add Val Loss as a line plot
    fig.add_trace(
        go.Scatter(
            x=df['Epoch'],
            y=df['Val Loss'],
            mode='lines+markers',
            name='Val Loss',
            line=dict(color='red', width=2),
            yaxis='y1'
        )
    )

    # Add Val Accuracy as a line plot with a secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df['Epoch'],
            y=df['Val Accuracy'],
            mode='lines+markers',
            name='Val Accuracy',
            line=dict(color='green', width=2),
            yaxis='y2'
        )
    )

    # Update layout with dual y-axes
    fig.update_layout(
        title=dict(
            text='ViT-Base-16-ImageNet21k training visualization',
            font=dict(size=26)
        ),
        xaxis=dict(
            title='Epoch',
            titlefont=dict(size=22),
            tickfont=dict(size=20)
        ),
        yaxis=dict(
            title='Loss',
            titlefont=dict(size=22),
            tickfont=dict(size=20),
            showgrid=True,
            zeroline=False
        ),
        yaxis2=dict(
            title='Validation Accuracy',
            titlefont=dict(size=22),
            tickfont=dict(size=20),
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False,
            range=[0.95, 1.0]
        ),
        legend=dict(
            x=0.01, y=0.99,
            font=dict(size=20)
        ),
        bargap=0.2,
        template='plotly_white'
    )

    fig.show()

# Example usage
# best_base_model_path = os.path.join(base_ViT_path, "Set2", "results.xlsx")
# base_model_results = pd.read_excel(best_base_model_path)
# plot_model_metrics(base_model_results)

# best_C100_PT_model_path = os.path.join(PT_C100_path, "FT_Set4", "Finetuning_results.xlsx")
# best_C100_PT_model_results = pd.read_excel(best_C100_PT_model_path)
# plot_model_metrics(best_C100_PT_model_results)

large_base_model_path = os.path.join(large_ViT_path, "Finetuning_results_grad_clipping_wd.xlsx")
large_base_model_results = pd.read_excel(large_base_model_path, nrows=50)
plot_model_metrics(large_base_model_results)



