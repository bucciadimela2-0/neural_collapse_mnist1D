from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nc_metrics import compute_nc_metrics
from utils.ProgressBar import ProgressBar


def train(x_train, y_train, x_test, y_test, model, 
          optimizer_name='SGD', num_epochs=200, eval_every=1, device='cpu'):
   
    #Training loop con supporto per diversi ottimizzatori
    
    
    
    train_ds = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(x_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, 128, shuffle=True)
    test_loader = DataLoader(test_ds, 256, shuffle=False)

    model = model.to(device)
    
   #Some set ups
    optimizer = get_optimizer(model, optimizer_name)
    criterion = nn.CrossEntropyLoss()
    scheduler = get_scheduler(optimizer, optimizer_name)
    
    history = defaultdict(list)
    tpt_start_epoch = None
    progress_bar = ProgressBar(num_epochs=num_epochs)

   
    print(f"Training with {optimizer_name}")
    for epoch in range(num_epochs):
       
        if optimizer_name == 'LBFGS':
            train_loss, train_acc = train_epoch_lbfgs(
                model, train_loader, optimizer, criterion, device
            )
        else:
            train_loss, train_acc = train_epoch_standard(
                model, train_loader, optimizer, criterion, device
            )
        
        if scheduler is not None:
            scheduler.step()
        
        # tpt detection
        if train_acc >= 0.999 and tpt_start_epoch is None:
            tpt_start_epoch = epoch
            print(f"\n🎯 TPT STARTS @ Epoch {epoch}\n")
        
        # eval
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            val_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            # Compute NC metrics 
            if epoch % max(eval_every, 1) == 0 or epoch == num_epochs - 1:
                metrics, H, means = compute_nc_metrics(model, test_loader, device)
                for k, v in metrics.items():
                    history[k].append(v)
                nc1, nc2, nc3, nc4 = (metrics['NC1_collapse'], metrics['NC2_etf'], 
                                      metrics['NC3_duality'], metrics['NC4_ncc'])
            else:
                nc1 = nc2 = nc3 = nc4 = None
            
            # Save history
            history['epoch'].append(epoch)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Update progress bar
            progress_bar.update(
                epoch=epoch,
                train_acc=train_acc,
                test_acc=test_acc,
                train_loss=train_loss,
                val_loss=val_loss,
                nc1=nc1, nc2=nc2, nc3=nc3, nc4=nc4
            )

    # final evaluation
    final_metrics, H_final, means_final = compute_nc_metrics(model, test_loader, device)
    
    return model, history, final_metrics, H_final, means_final, tpt_start_epoch


# ============================================================================
# OPTIMIZER FACTORY
# ============================================================================
def get_optimizer(model, optimizer_name):
    #create optimizer
    optimizers = {
        'SGD': lambda: optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4),
        'Adam': lambda: optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4),
        'LBFGS': lambda: optim.LBFGS(model.parameters(), lr=0.1, max_iter=20, history_size=10),
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Optimizer {optimizer_name} not supported. Choose from {list(optimizers.keys())}")
    
    return optimizers[optimizer_name]()


def get_scheduler(optimizer, optimizer_name):
    #create learning rate scheduler
    if optimizer_name == 'LBFGS':
        return None  # LBFGS uses line search
    else:
        return optim.lr_scheduler.StepLR(optimizer, step_size=117, gamma=0.1)


#training epochs
def train_epoch_standard(model, loader, optimizer, criterion, device):
    #Standard training epoch for first-order methods
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_epoch_lbfgs(model, loader, optimizer, criterion, device):
    #Training epoch for L-BFGS which requires closure
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        
        def closure():
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        
        with torch.no_grad():
            logits = model(xb)
            total_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    #Evaluation on test set
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            
            total_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy