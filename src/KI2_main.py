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
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F


current_dir = os.getcwd()
base_ViT_path = os.path.join(current_dir, "models", "base_ViT")
base_ViT_intermediate_HP_path = os.path.join(base_ViT_path, "HPT_intermediate_testing.xlsx")
base_ViT_Set1_path = os.path.join(current_dir, "models", "base_ViT", "Set1")
base_ViT_Set2_path = os.path.join(current_dir, "models", "base_ViT", "Set2")
base_ViT_Set3_path = os.path.join(current_dir, "models", "base_ViT", "Set3")
base_ViT_Set4_path = os.path.join(current_dir, "models", "base_ViT", "Set4")
base_ViT_Set5_path = os.path.join(current_dir, "models", "base_ViT", "Set5")

PT_CIFAR100_ViT_path = os.path.join(current_dir, "models", "PT_CIFAR100_ViT")
PT_CIFAR100_ViT_intermediate_HP_path = os.path.join(PT_CIFAR100_ViT_path, "HPT_intermediate_testing.xlsx")
PT_CIFAR100_Set1_path = os.path.join(PT_CIFAR100_ViT_path, "Set1")
PT_CIFAR100_Set2_path = os.path.join(PT_CIFAR100_ViT_path, "Set2")
PT_CIFAR100_Set3_path = os.path.join(PT_CIFAR100_ViT_path, "Set3")
PT_CIFAR100_Set4_path = os.path.join(PT_CIFAR100_ViT_path, "Set4")
PT_CIFAR100_Set5_path = os.path.join(PT_CIFAR100_ViT_path, "Set5")

FT_CIFAR100_Set1_path = os.path.join(PT_CIFAR100_ViT_path, "FT_Set1")
FT_CIFAR100_Set2_path = os.path.join(PT_CIFAR100_ViT_path, "FT_Set2")
FT_CIFAR100_Set3_path = os.path.join(PT_CIFAR100_ViT_path, "FT_Set3")
FT_CIFAR100_Set4_path = os.path.join(PT_CIFAR100_ViT_path, "FT_Set4")
FT_CIFAR100_Set5_path = os.path.join(PT_CIFAR100_ViT_path, "FT_Set5")


# trainloader100, testloader100 = get_cifar100_data()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(0)}")

# Get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# Move to the device (CUDA)
# images, labels = images.to(device), labels.to(device)
"""
Part 1 : No pretraining at all
"""
def study():
    def HPT_finetuning (trial):
        
        # Early testing 
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)  
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)    
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)               
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])        
        num_layers = trial.suggest_categorical('num_layers', [6, 12])              
        num_heads = trial.suggest_categorical('num_heads', [8, 16])            
        emb_size = trial.suggest_categorical('emb_size', [256, 512, 768])
        ff_hidden_size = trial.suggest_categorical('ff_hidden_size', [512, 1024, 2048])
        
        # Intermediate testing
        learning_rate = trial.suggest_float('learning_rate', 5e-5, 5e-4, log=True)  
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)    
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)               
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])        
        num_layers = trial.suggest_categorical('num_layers', [12])              
        num_heads = trial.suggest_categorical('num_heads', [8, 16])            
        emb_size = trial.suggest_categorical('emb_size', [512, 768])
        ff_hidden_size = trial.suggest_categorical('ff_hidden_size', [512, 1024, 2048])
        
        
        trainloader10, testloader10 = get_cifar10_data(batch_size)  #Load in Data
        
        # Initialize model
        model = VisionTransformer(
            num_classes=10,
            num_layers=num_layers,
            num_heads=num_heads,
            emb_size=emb_size,
            dropout=dropout_rate,
            ff_hidden_size=ff_hidden_size
        )
        
        model.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
        # Training loop
        for epoch in range(30):
            train_loss = train(model, trainloader10, criterion, optimizer, device)
            val_loss, val_accuracy = validate(model, testloader10, criterion, device)
            
            # Report the validation loss to Optuna
            trial.report(val_loss, epoch)
            
            
            # Pruning unpromising trials
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            scheduler.step()
            
        trial.set_user_attr('train_loss', train_loss)     
        trial.set_user_attr('validation_accuracy', val_accuracy)
    
        return val_loss
    
    study = optuna.create_study(direction="minimize")
    study.optimize(HPT_finetuning, n_trials=25, callbacks=[progress_callback])  # Number of trials
    
    results = []
    for trial in study.trials:
        trial_result = trial.params  # Extract the hyperparameters
        trial_result['value'] = trial.value  # Extract the objective value (e.g., validation loss)
        trial_result['validation_accuracy'] = trial.user_attrs.get('validation_accuracy', None)
        trial_result['number'] = trial.number  # Trial number
        
        trial_result['datetime_start'] = trial.datetime_start  # Start time of the trial
        trial_result['datetime_complete'] = trial.datetime_complete  # Completion time of the trial
        results.append(trial_result)
    
    df_results = pd.DataFrame(results)
    
    
    HPT_base_path = os.path.join(base_ViT_path, "HPT_intermediate_testing.xlsx")
    df_results.to_excel(HPT_base_path, index=False)
    print("Best hyperparameters found were: ", study.best_params)
    

def base_ViT(HP, model_path, num_epochs=100): 
    model_save_path = os.path.join(model_path, "base_ViT.pth")
    
    batch_size = int(HP["batch_size"])
    learning_rate = HP["learning_rate"]
    weight_decay = HP["weight_decay"]
    dropout_rate = HP["dropout_rate"]
    num_layers = HP["num_layers"]
    num_heads = HP["num_heads"]
    emb_size = HP["emb_size"]
    ff_hidden_size = HP["ff_hidden_size"]

    trainloader10, testloader10 = get_cifar10_data(batch_size)
    
    model = VisionTransformer(
        num_classes=10,
        num_layers=num_layers,
        num_heads=num_heads,
        emb_size=emb_size,
        dropout=dropout_rate,
        ff_hidden_size=ff_hidden_size
    )
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    
    results_list = []
    best_acc = 0
    for epoch in range(num_epochs):
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
            torch.save(model.state_dict(), model_save_path)
            best_acc = val_accuracy
        
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(model_path, "results.xlsx")
    results_df.to_excel(results_path, index=False)
    
    
base_HP_df = pd.read_excel(base_ViT_intermediate_HP_path)
base_HP_df = base_HP_df.sort_values(by="value", ascending=True)
base_HP_Set1 = base_HP_df.iloc[0]
base_HP_Set2 = base_HP_df.iloc[1]
base_HP_Set3 = base_HP_df.iloc[2]
base_HP_Set4 = base_HP_df.iloc[3]
base_HP_Set5 = base_HP_df.iloc[4]


# base_ViT(base_HP_Set1, base_ViT_Set1_path, num_epochs=100)
# base_ViT(base_HP_Set2, base_ViT_Set2_path, num_epochs=100)
# base_ViT(base_HP_Set3, base_ViT_Set3_path, num_epochs=100)
# base_ViT(base_HP_Set4, base_ViT_Set4_path, num_epochs=100)
# base_ViT(base_HP_Set5, base_ViT_Set5_path, num_epochs=100)


def ensemble_acc():
    base_model1_path = os.path.join(base_ViT_Set1_path, "base_ViT.pth")
    base_model2_path = os.path.join(base_ViT_Set2_path, "base_ViT.pth")
    base_model3_path = os.path.join(base_ViT_Set3_path, "base_ViT.pth")
    base_model4_path = os.path.join(base_ViT_Set4_path, "base_ViT.pth")
    base_model5_path = os.path.join(base_ViT_Set5_path, "base_ViT.pth")
    
    base_model1 = VisionTransformer(num_classes=10,
                                    dropout = base_HP_Set1["dropout_rate"],
                                    num_layers = base_HP_Set1["num_layers"],
                                    num_heads = base_HP_Set1["num_heads"],
                                    emb_size = base_HP_Set1["emb_size"],
                                    ff_hidden_size = base_HP_Set1["ff_hidden_size"]
                                    )
    base_model1.load_state_dict(torch.load(base_model1_path))
    
    base_model2 = VisionTransformer(num_classes=10,
                                    dropout = base_HP_Set2["dropout_rate"],
                                    num_layers = base_HP_Set2["num_layers"],
                                    num_heads = base_HP_Set2["num_heads"],
                                    emb_size = base_HP_Set2["emb_size"],
                                    ff_hidden_size = base_HP_Set2["ff_hidden_size"]
                                    )
    base_model2.load_state_dict(torch.load(base_model2_path))
    
    base_model3 = VisionTransformer(num_classes=10,
                                    dropout = base_HP_Set3["dropout_rate"],
                                    num_layers = base_HP_Set3["num_layers"],
                                    num_heads = base_HP_Set3["num_heads"],
                                    emb_size = base_HP_Set3["emb_size"],
                                    ff_hidden_size = base_HP_Set3["ff_hidden_size"]
                                    )
    base_model3.load_state_dict(torch.load(base_model3_path))
    
    base_model4 = VisionTransformer(num_classes=10,
                                    dropout = base_HP_Set4["dropout_rate"],
                                    num_layers = base_HP_Set4["num_layers"],
                                    num_heads = base_HP_Set4["num_heads"],
                                    emb_size = base_HP_Set4["emb_size"],
                                    ff_hidden_size = base_HP_Set4["ff_hidden_size"]
                                    )
    base_model4.load_state_dict(torch.load(base_model4_path))
    
    base_model5 = VisionTransformer(num_classes=10,
                                    dropout = base_HP_Set5["dropout_rate"],
                                    num_layers = base_HP_Set5["num_layers"],
                                    num_heads = base_HP_Set5["num_heads"],
                                    emb_size = base_HP_Set5["emb_size"],
                                    ff_hidden_size = base_HP_Set5["ff_hidden_size"]
                                    )
    base_model5.load_state_dict(torch.load(base_model5_path))
    
    models = [base_model1, base_model2, base_model3, base_model4, base_model5]
    
    trainloader10, testloader10 = get_cifar10_data(32)
    all_outputs = []
    for model in models:
        model.to(device)
        model.eval()
        output_list = []
        with torch.no_grad():
            for inputs, labels in testloader10:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                # Apply softmax to get probabilities
                probs = F.softmax(outputs, dim=1)

                output_list.append(probs.cpu().numpy())
        all_outputs.append(np.concatenate(output_list, axis=0))
               
    # Step 2: Average predictions for ensemble output
    stacked_outputs = np.stack(all_outputs, axis=0)  # Shape: (num_models, num_samples, num_classes)
    ensemble_preds = np.mean(stacked_outputs, axis=0)  # Average across models, shape: (num_samples, num_classes)
    
    # Step 3: Determine final predicted classes
    ensemble_classes = np.argmax(ensemble_preds, axis=1)  # Get class with highest probability
    
    # Optional: Calculate ensemble accuracy
    test_labels = np.concatenate([labels.numpy() for _, labels in testloader10], axis=0)
    ensemble_accuracy = np.mean(ensemble_classes == test_labels) * 100

    print(f'Ensemble Accuracy: {ensemble_accuracy:.2f}%')
    return all_outputs, ensemble_preds
# outputs, ensemble_mean = ensemble_acc()
    

"""
Part 2: Pretraining on CIFAR-100
"""
def pretraining_study(trial):
    #Second Phase
    learning_rate = trial.suggest_float('learning_rate', 5e-5, 1e-1, log=True)  
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)    
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)               
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])        
    # num_layers = trial.suggest_categorical('num_layers', [12])              
    # num_heads = trial.suggest_categorical('num_heads', [8])            
    # emb_size = trial.suggest_categorical('emb_size', [768])
    ff_hidden_size = trial.suggest_categorical('ff_hidden_size', [2048])
    warmup_percentage = trial.suggest_float("warmup_steps", 0.05, 0.2)
    
    trainloader100, testloader100 = get_cifar100_data(batch_size)  #Load in Data
    
    # Initialize model
    model = VisionTransformer(
        num_classes=100,
        num_layers=12,
        num_heads=8,
        emb_size=768,
        dropout=dropout_rate,
        ff_hidden_size=ff_hidden_size
    )
    
    model.to(device)
    
    PT_optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_epoch = 10
    total_steps = 1562.5 * num_epoch
    warmup_steps = total_steps * warmup_percentage
    criterion = nn.CrossEntropyLoss()
    PT_schedular = linear_warmup_decay(PT_optimizer, warmup_steps=warmup_steps, total_steps=total_steps, lr_max=learning_rate)
    best_acc = 0
    # Training loop
    for epoch in range(num_epoch):
        train_loss = pretrain(model, trainloader100, criterion, PT_optimizer, PT_schedular, device)
        val_loss, val_accuracy = validate(model, testloader100, criterion, device)
        
        # Report the validation loss to Optuna
        trial.report(val_loss, epoch)
        
        
        # Pruning unpromising trials
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            

    trial.set_user_attr('train_loss', train_loss)     
    trial.set_user_attr('validation_accuracy', val_accuracy)

    return val_loss

# study = optuna.create_study(direction="minimize")
# study.optimize(pretraining_study, n_trials=25, callbacks=[progress_callback])  # Number of trials

# results = []
# for trial in study.trials:
#     trial_result = trial.params  # Extract the hyperparameters
#     trial_result['value'] = trial.value  # Extract the objective value (e.g., validation loss)
#     trial_result['validation_accuracy'] = trial.user_attrs.get('validation_accuracy', None)
#     trial_result['number'] = trial.number  # Trial number
    
#     trial_result['datetime_start'] = trial.datetime_start  # Start time of the trial
#     trial_result['datetime_complete'] = trial.datetime_complete  # Completion time of the trial
#     results.append(trial_result)

# df_results = pd.DataFrame(results)


# HPT_pretraining_path = os.path.join(PT_CIFAR100_ViT_path, "HPT_intermediate_testing3.xlsx")
# df_results.to_excel(HPT_pretraining_path, index=False)
# print("Best hyperparameters found were: ", study.best_params)


def PT_C100_ViT(HP, model_path, num_epoch):
    model_save_path = os.path.join(model_path, "base_ViT.pth")
    
    batch_size = int(HP["batch_size"])
    learning_rate = HP["learning_rate"]
    weight_decay = HP["weight_decay"]
    dropout_rate = HP["dropout_rate"]
    ff_hidden_size = HP["ff_hidden_size"]
    warmup_percentage = HP["warmup_steps"]
    
    trainloader100, testloader100 = get_cifar100_data(batch_size)
    
    model = VisionTransformer(
        num_classes=100,
        num_layers=12,
        num_heads=8,
        emb_size=768,
        dropout=dropout_rate,
        ff_hidden_size=ff_hidden_size
    )
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    total_steps = 1562.5 * num_epoch
    warmup_steps = total_steps * warmup_percentage
    criterion = nn.CrossEntropyLoss()
    scheduler = linear_warmup_decay(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, lr_max=learning_rate)
    best_acc = 0
    
    results_list = []
    best_acc = 0
    for epoch in range(num_epoch):
        start_time = time.time()
        
        train_loss = pretrain(model, trainloader100, criterion, optimizer, scheduler, device)
        val_loss, val_accuracy = validate(model, testloader100, criterion, device)
        
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
           
        if val_accuracy > best_acc:
            torch.save(model.state_dict(), model_save_path)
            best_acc = val_accuracy
        
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(model_path, "results.xlsx")
    results_df.to_excel(results_path, index=False)

C100_HP_df = pd.read_excel(PT_CIFAR100_ViT_intermediate_HP_path)
C100_HP_df = C100_HP_df.sort_values(by="value", ascending=True)
C100_HP_Set1 = C100_HP_df.iloc[0]
C100_HP_Set2 = C100_HP_df.iloc[1]
C100_HP_Set3 = C100_HP_df.iloc[2]
C100_HP_Set4 = C100_HP_df.iloc[3]
C100_HP_Set5 = C100_HP_df.iloc[4]


# PT_C100_ViT(C100_HP_Set1, PT_CIFAR100_Set1_path, num_epoch=100)
# PT_C100_ViT(C100_HP_Set2, PT_CIFAR100_Set2_path, num_epoch=100)
# PT_C100_ViT(C100_HP_Set3, PT_CIFAR100_Set3_path, num_epoch=100)
# PT_C100_ViT(C100_HP_Set4, PT_CIFAR100_Set4_path, num_epoch=100)
# PT_C100_ViT(C100_HP_Set5, PT_CIFAR100_Set5_path, num_epoch=100)

def best_model(base_path):
    best_results = pd.DataFrame()
    
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if "Set" in dir_name:
                folder_path = os.path.join(root, dir_name)
                results_file = os.path.join(folder_path, 'results.xlsx')
                
                if os.path.exists(results_file):
                    try:
                        df = pd.read_excel(results_file)
                        if 'Val Accuracy' in df.columns:
                            best_row = df.loc[df["Val Accuracy"].idxmax()]
                            best_results[dir_name] = best_row
                    except Exception as e:
                        print(f"Error loading {results_file}: {e}")
    
    return best_results

PT_model_comparism = best_model(PT_CIFAR100_ViT_path)

"""
Finetuning on CIFAR-10
"""
def finetuning_study(trial):
    best_model_path = os.path.join(PT_CIFAR100_Set5_path, "base_ViT.pth") 
    PT_model = VisionTransformer(num_classes=100,
                                    num_layers=12,
                                    num_heads=8,
                                    emb_size=768,
                                    dropout=C100_HP_Set5["dropout_rate"],
                                    ff_hidden_size=C100_HP_Set5["ff_hidden_size"]
                                    )
    
    PT_model.load_state_dict(torch.load(best_model_path))
    PT_model.mlp_head = nn.Sequential(              # Reinitialise the output layer
        nn.LayerNorm(PT_model.mlp_head[0].normalized_shape),
        nn.Linear(PT_model.mlp_head[1].in_features, 10)
        )
    nn.init.zeros_(PT_model.mlp_head[1].weight)     #Set all weights/biases to 0 for new output layer
    nn.init.zeros_(PT_model.mlp_head[1].bias)
    PT_model.to(device)
    
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)  
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)                  
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])        
    
    trainloader10, testloader10 = get_cifar10_data(batch_size)  #Load in Data
    
    optimizer = optim.AdamW(PT_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=25)

    # Training loop
    for epoch in range(25):
        train_loss = train(PT_model, trainloader10, criterion, optimizer, device)
        val_loss, val_accuracy = validate(PT_model, testloader10, criterion, device)
        
        # Report the validation loss to Optuna
        trial.report(val_loss, epoch)
        
        
        # Pruning unpromising trials
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
             
            
        scheduler.step()
        
    trial.set_user_attr('train_loss', train_loss)     
    trial.set_user_attr('validation_accuracy', val_accuracy)

    return val_loss

# study = optuna.create_study(direction="minimize")
# study.optimize(finetuning_study, n_trials=20, callbacks=[progress_callback])  # Number of trials

# results = []
# for trial in study.trials:
#     trial_result = trial.params  # Extract the hyperparameters
#     trial_result['value'] = trial.value  # Extract the objective value (e.g., validation loss)
#     trial_result['validation_accuracy'] = trial.user_attrs.get('validation_accuracy', None)
#     trial_result['number'] = trial.number  # Trial number
    
#     trial_result['datetime_start'] = trial.datetime_start  # Start time of the trial
#     trial_result['datetime_complete'] = trial.datetime_complete  # Completion time of the trial
#     results.append(trial_result)

# df_results = pd.DataFrame(results)


# HPT_finetuning_path = os.path.join(PT_CIFAR100_ViT_path, "HPT_finetuning_intermediate_testing.xlsx")
# df_results.to_excel(HPT_finetuning_path, index=False)
# print("Best hyperparameters found were: ", study.best_params)

HPT_finetuning_path = os.path.join(PT_CIFAR100_ViT_path, "HPT_finetuning_intermediate_testing.xlsx")
FT_C100_HP_df = pd.read_excel(HPT_finetuning_path)
FT_C100_HP_df = FT_C100_HP_df.sort_values(by="value", ascending=True)
FT_C100_HP_Set1 = FT_C100_HP_df.iloc[0]
FT_C100_HP_Set2 = FT_C100_HP_df.iloc[1]
FT_C100_HP_Set3 = FT_C100_HP_df.iloc[2]
FT_C100_HP_Set4 = FT_C100_HP_df.iloc[3]
FT_C100_HP_Set5 = FT_C100_HP_df.iloc[4]

def FT_C100_ViT(HP_Set_Finetuning, model_path, num_epochs):
    """
    Finetuning
    """
    best_PT_model_path = os.path.join(PT_CIFAR100_Set5_path, "base_ViT.pth")
    PT_model = VisionTransformer(num_classes=100,
                                    num_layers=12,
                                    num_heads=8,
                                    emb_size=768,
                                    dropout=C100_HP_Set5["dropout_rate"],
                                    ff_hidden_size=C100_HP_Set5["ff_hidden_size"]
                                    )
    PT_model.load_state_dict(torch.load(best_PT_model_path))
    PT_model.mlp_head = nn.Sequential(              # Reinitialise the output layer
        nn.LayerNorm(PT_model.mlp_head[0].normalized_shape),
        nn.Linear(PT_model.mlp_head[1].in_features, 10)
        )
    nn.init.zeros_(PT_model.mlp_head[1].weight)     #Set all weights/biases to 0 for new output layer
    nn.init.zeros_(PT_model.mlp_head[1].bias)
    PT_model.to(device)
    
    learning_rate = HP_Set_Finetuning["learning_rate"]
    weight_decay = HP_Set_Finetuning["weight_decay"]
    batch_size = int(HP_Set_Finetuning["batch_size"])
    
    optimizer = optim.AdamW(PT_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    schedular = CosineAnnealingLR(optimizer, T_max=num_epochs)   
    
    trainloader10, testloader10 = get_cifar10_data(batch_size)  #Load in Data
    
    model_save_path = os.path.join(model_path, "C100_PT_ViT.pth")
    results_list = []
    best_acc = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train(PT_model, trainloader10, criterion, optimizer, device)
        val_loss, val_accuracy = validate(PT_model, testloader10, criterion, device)
        
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
        
        if val_accuracy > best_acc:
            torch.save(PT_model.state_dict(), model_save_path)
            best_acc = val_accuracy
            
        schedular.step()
    
 
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(model_path, "Finetuning_results.xlsx")
    results_df.to_excel(results_path, index=False)
    
    return PT_model
    
# Finetuned_C100_PT_ViT1 = FT_C100_ViT(FT_C100_HP_Set1, FT_CIFAR100_Set1_path, num_epochs=30)
# Finetuned_C100_PT_ViT2 = FT_C100_ViT(FT_C100_HP_Set2, FT_CIFAR100_Set2_path, num_epochs=30)
# Finetuned_C100_PT_ViT3 = FT_C100_ViT(FT_C100_HP_Set3, FT_CIFAR100_Set3_path, num_epochs=30)
# Finetuned_C100_PT_ViT4 = FT_C100_ViT(FT_C100_HP_Set4, FT_CIFAR100_Set4_path, num_epochs=30)
# Finetuned_C100_PT_ViT5 = FT_C100_ViT(FT_C100_HP_Set5, FT_CIFAR100_Set5_path, num_epochs=30)

def ensemble_acc_PT_ViT():
    base_model1_path = os.path.join(FT_CIFAR100_Set1_path, "C100_PT_ViT.pth")
    base_model2_path = os.path.join(FT_CIFAR100_Set2_path, "C100_PT_ViT.pth")
    base_model3_path = os.path.join(FT_CIFAR100_Set3_path, "C100_PT_ViT.pth")
    base_model4_path = os.path.join(FT_CIFAR100_Set4_path, "C100_PT_ViT.pth")
    base_model5_path = os.path.join(FT_CIFAR100_Set5_path, "C100_PT_ViT.pth")
    
    base_model1 = VisionTransformer(num_classes=10,
                                    dropout = C100_HP_Set1["dropout_rate"],
                                    num_layers = 12,
                                    num_heads = 8,
                                    emb_size = 768,
                                    ff_hidden_size = C100_HP_Set1["ff_hidden_size"]
                                    )
    base_model1.load_state_dict(torch.load(base_model1_path))
    
    base_model2 = VisionTransformer(num_classes=10,
                                    dropout = C100_HP_Set2["dropout_rate"],
                                    num_layers = 12,
                                    num_heads = 8,
                                    emb_size = 768,
                                    ff_hidden_size = C100_HP_Set2["ff_hidden_size"]
                                    )
    base_model2.load_state_dict(torch.load(base_model2_path))
    
    base_model3 = VisionTransformer(num_classes=10,
                                    dropout = C100_HP_Set3["dropout_rate"],
                                    num_layers = 12,
                                    num_heads = 8,
                                    emb_size = 768,
                                    ff_hidden_size = C100_HP_Set3["ff_hidden_size"]
                                    )
    base_model3.load_state_dict(torch.load(base_model3_path))
    
    base_model4 = VisionTransformer(num_classes=10,
                                    dropout = C100_HP_Set4["dropout_rate"],
                                    num_layers = 12,
                                    num_heads = 8,
                                    emb_size = 768,
                                    ff_hidden_size = C100_HP_Set4["ff_hidden_size"]
                                    )
    base_model4.load_state_dict(torch.load(base_model4_path))
    
    base_model5 = VisionTransformer(num_classes=10,
                                    dropout = C100_HP_Set5["dropout_rate"],
                                    num_layers = 12,
                                    num_heads = 8,
                                    emb_size = 768,
                                    ff_hidden_size = C100_HP_Set5["ff_hidden_size"]
                                    )
    base_model5.load_state_dict(torch.load(base_model5_path))
    
    models = [base_model1, base_model2, base_model3, base_model4, base_model5]
    
    trainloader10, testloader10 = get_cifar10_data(32)
    all_outputs = []
    for model in models:
        model.to(device)
        model.eval()
        output_list = []
        with torch.no_grad():
            for inputs, labels in testloader10:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                # Apply softmax to get probabilities
                probs = F.softmax(outputs, dim=1)

                output_list.append(probs.cpu().numpy())
        all_outputs.append(np.concatenate(output_list, axis=0))
               
    # Step 2: Average predictions for ensemble output
    stacked_outputs = np.stack(all_outputs, axis=0)  # Shape: (num_models, num_samples, num_classes)
    ensemble_preds = np.mean(stacked_outputs, axis=0)  # Average across models, shape: (num_samples, num_classes)
    
    # Step 3: Determine final predicted classes
    ensemble_classes = np.argmax(ensemble_preds, axis=1)  # Get class with highest probability
    
    # Optional: Calculate ensemble accuracy
    test_labels = np.concatenate([labels.numpy() for _, labels in testloader10], axis=0)
    ensemble_accuracy = np.mean(ensemble_classes == test_labels) * 100

    print(f'Ensemble Accuracy: {ensemble_accuracy:.2f}%')
    return all_outputs, ensemble_preds

outputs, ensemble_mean = ensemble_acc_PT_ViT()

    

