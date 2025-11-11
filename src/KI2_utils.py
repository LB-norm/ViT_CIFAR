import optuna
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

def get_cifar10_data(b_size):
    # Transform Pipeline 
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=False, num_workers=0)
    # image, label = testset[24]  # Index 2 for the 3rd image

    # # Convert the image tensor to a numpy array and display it
    # # If your transform includes normalization or other tensor operations, you might need to reverse them for display
    # # Plotting the image in its original size (32x32 pixels)
    # image = image.permute(1, 2, 0)
    # plt.figure(figsize=(1, 1))  # Set figure size to (1, 1) inch to match 32x32 pixels
    # plt.imshow(image)
    # plt.title(f'Label: {label}')  # Optionally display the label
    # plt.axis('off')
    # plt.show()
    return trainloader, testloader

def get_cifar10_for_pretrained_VIT(feature_extractor, batch_size):
    train_transform = transforms.Compose([
        Resize((224, 224)),  # Resize images to 224x224 pixels
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.2),
        ToTensor(),
        Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    
    val_transform = transforms.Compose([
        Resize((224, 224)),  # Resize images to 224x224 pixels
        ToTensor(),
        Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return trainloader, testloader
    
    
def get_cifar100_data(b_size):
    # Transform Pipeline 
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=False, num_workers=0)
    
    return trainloader, testloader

def linear_warmup_decay(optimizer, warmup_steps, total_steps, lr_max):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def pretrain(model, trainloader, criterion, optimizer, lwd_schedular, device):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        lwd_schedular.step()
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(trainloader.dataset)
    return epoch_loss

def train(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Handle different output types (e.g., ImageClassifierOutput)
        if isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
            
        loss = criterion(logits, labels)
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(trainloader.dataset)
    return epoch_loss

def validate(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            # Handle different output types (e.g., ImageClassifierOutput)
            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(testloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

# Custom callback function for progress reporting
def progress_callback(study, trial):
    best_trial = study.best_trial
    print(f"Trial {len(study.trials)} completed.")
    print(f"Best Validation Loss so far: {best_trial.value:.4f}")
    if 'validation_accuracy' in best_trial.user_attrs:
        print(f"Best Validation Accuracy so far: {best_trial.user_attrs['validation_accuracy']:.4f}")
    print("-" * 50)