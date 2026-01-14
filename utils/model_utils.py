from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nc_metrics import compute_layer_nc_metrics, compute_nc_metrics
from utils.ProgressBar import ProgressBar


def train(
    x_train, y_train, x_test, y_test,
    model,
    optimizer_name="SGD",
    num_epochs=200,
    eval_every=1,
    device="cpu",
    layer_names=None,              # <-- NEW: abilita layer-wise se lista
    layer_eval_every=None,         # <-- NEW: frequenza layer-wise (default=eval_every)
):
    """
    Training loop con supporto per diversi ottimizzatori.

    Se layer_names è None:
        - calcola NC classico (penultimate) con compute_nc_metrics
        - history: defaultdict(list)

    Se layer_names è lista:
        - calcola NC per layer con extract_layer_features + compute_layer_nc_metrics
        - history_global: info globali
        - history_layerwise: NC per layer
    """

    train_ds = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
    test_ds  = TensorDataset(torch.FloatTensor(x_test),  torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, 128, shuffle=True)
    test_loader  = DataLoader(test_ds, 256, shuffle=False)

    model = model.to(device)

    optimizer = get_optimizer(model, optimizer_name)
    criterion = nn.CrossEntropyLoss()
    scheduler = get_scheduler(optimizer, optimizer_name)

    # ----- histories -----
    history = defaultdict(list)  # sempre presente per compatibilità
    history_layerwise = None
    if layer_names is not None:
        history_layerwise = defaultdict(lambda: defaultdict(list))
        history_layerwise["global"] = defaultdict(list)

    tpt_start_epoch = None
    progress_bar = ProgressBar(num_epochs=num_epochs)

    if layer_eval_every is None:
        layer_eval_every = eval_every

    print(f"Training with {optimizer_name}")

    for epoch in range(num_epochs):

        # ========== TRAINING ==========
        if optimizer_name == "LBFGS":
            train_loss, train_acc = train_epoch_lbfgs(model, train_loader, optimizer, criterion, device)
        else:
            train_loss, train_acc = train_epoch_standard(model, train_loader, optimizer, criterion, device)

        if scheduler is not None:
            scheduler.step()

        # ========== TPT DETECTION ==========
        if train_acc >= 0.999 and tpt_start_epoch is None:
            tpt_start_epoch = epoch
            print(f"\n🎯 TPT STARTS @ Epoch {epoch}\n")

        # ========== EVALUATION ==========
        if epoch % eval_every == 0 or epoch == num_epochs - 1:
            val_loss, test_acc = evaluate(model, test_loader, criterion, device)

            # ---- salva sempre history "base" (compatibilità) ----
            history["epoch"].append(epoch)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # ======================================================
            # NC COMPUTATION: classico (penultimate) O layer-wise
            # ======================================================
            nc1 = nc2 = nc3 = nc4 = None

            # ---- Caso A: NC classico ----
            if layer_names is None:
                metrics, H, means = compute_nc_metrics(model, test_loader, device)

                for k, v in metrics.items():
                    history[k].append(v)

                nc1, nc2, nc3, nc4 = (
                    metrics.get("NC1_collapse"),
                    metrics.get("NC2_etf"),
                    metrics.get("NC3_duality"),
                    metrics.get("NC4_ncc"),
                )

            # ---- Caso B: Layer-wise NC ----
            else:
                # calcola layer-wise solo ogni layer_eval_every (per velocità)
                if (epoch % layer_eval_every == 0) or (epoch == num_epochs - 1):

                    layer_features, labels = extract_layer_features(
                        model=model,
                        loader=test_loader,
                        device=device,
                        layer_names=layer_names
                    )

                    layer_metrics = compute_layer_nc_metrics(layer_features, labels)

                    # salva global anche in history_layerwise
                    hg = history_layerwise["global"]
                    hg["epoch"].append(epoch)
                    hg["train_acc"].append(train_acc)
                    hg["test_acc"].append(test_acc)
                    hg["train_loss"].append(train_loss)
                    hg["val_loss"].append(val_loss)

                    # salva per layer (+ epoch per plotting corretto)
                    for lname in layer_names:
                        history_layerwise[lname]["epoch"].append(epoch)
                        m = layer_metrics.get(lname, {})
                        for mk, mv in m.items():
                            history_layerwise[lname][mk].append(mv)

                    # per progress bar mostro penultimate se esiste
                    pen = layer_metrics.get("penultimate", layer_metrics.get("feature_transform", {}))
                    nc1 = pen.get("NC1", None)
                    nc2 = pen.get("NC2", None)
                    nc3 = pen.get("NC3", None)
                    nc4 = pen.get("NC4", None)

            # ---- Update progress bar ----
            progress_bar.update(
                epoch=epoch,
                train_acc=train_acc,
                test_acc=test_acc,
                train_loss=train_loss,
                val_loss=val_loss,
                nc1=nc1, nc2=nc2, nc3=nc3, nc4=nc4,
            )

    # ========== FINAL EVALUATION ==========
    final_metrics, H_final, means_final = compute_nc_metrics(model, test_loader, device)

    # ritorni:
    # - se layerwise: aggiungo history_layerwise come output extra
    if history_layerwise is not None:
        return model, history, history_layerwise, final_metrics, H_final, means_final, tpt_start_epoch

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

def extract_layer_features(model, loader, device, layer_names):
    """Extract features from all specified layers"""
    model.eval()
    layer_features = {name: [] for name in layer_names}
    labels_list = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            _ = model(xb)
            
            for layer_name in layer_names:
                layer_features[layer_name].append(
                    model.layer_features[layer_name].cpu()
                )
            labels_list.append(yb)
    
    # Concatenate
    for layer_name in layer_names:
        layer_features[layer_name] = torch.cat(layer_features[layer_name]).numpy()
    
    labels = torch.cat(labels_list).numpy()
    
    return layer_features, labels