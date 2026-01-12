import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def compute_nc1(H, Y, num_classes):
    #Compute NC1: Tr(Σ_W) / Tr(Σ_T)

    mu_G = H.mean(axis=0, keepdims=True)
    H_centered = H - mu_G
    Sigma_T = (H_centered.T @ H_centered) / len(H)

    Sigma_W = np.zeros_like(Sigma_T)
    for k in range(num_classes):
        H_k = H[Y == k]
        if len(H_k) > 0:
            mu_k = H_k.mean(axis=0, keepdims=True)
            H_k_centered = H_k - mu_k
            Sigma_W += (H_k_centered.T @ H_k_centered) / len(H)

    nc1 = np.trace(Sigma_W) / (np.trace(Sigma_T) + 1e-8)
    return nc1

def compute_nc2(H, Y, num_classes):
    
    #NC2: Convergence to Simplex ETF
    #Class means form equiangular tight frame: <μ_i, μ_j> = -1/(K-1) for i≠j
    
    # Global mean
    mu_G = H.mean(axis=0, keepdims=True)

    # Class means, centered and normalized
    class_means = np.stack([H[Y == k].mean(0) for k in range(num_classes)])
    means_centered = class_means - mu_G
    means_norm = means_centered / (np.linalg.norm(means_centered, axis=1, keepdims=True) + 1e-8)

    # Gram matrix off-diagonal
    gram = means_norm @ means_norm.T
    np.fill_diagonal(gram, 0)

    # ETF ideal: -1/(K-1)
    ideal_etf = -1.0 / (num_classes - 1)
    nc2 = np.mean(np.abs(gram - ideal_etf))

    return nc2, means_norm

def compute_nc3(model, means_norm, num_classes):
    
    #NC3: Self-duality (W ≈ M^T)
    # Classifier weights collapse to class means (up to rescaling)
    
    # Classifier weights [d × K]
    W = model.classifier.weight.data.cpu().numpy().T
    W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)

    # Frobenius norm distance
    nc3 = np.linalg.norm(W_norm - means_norm.T, 'fro') / num_classes
    return nc3

def compute_nc4(H, means_norm, Y):
    
    # NC4: Nearest Class Center decision rule
    # argmin_c ||h(x) - μ_c||_2 dominates classification
    
    # Normalize features
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

    # Euclidean distances to class means
    dists = np.linalg.norm(H_norm[:, None] - means_norm[None, :], axis=2)
    ncc_pred = np.argmin(dists, axis=1)

    # NCC accuracy
    nc4 = np.mean(ncc_pred == Y)
    return nc4

def compute_nc5(H, Y, num_classes):
    
    #NC5: Simplex volume collapse
    #Class means collapse to lower-dimensional simplex (normalized log-det)
    
    # Global mean
    mu_G = H.mean(axis=0, keepdims=True)

    # Centered class means
    class_means = np.stack([H[Y == k].mean(0) for k in range(num_classes)])
    means_centered = class_means - mu_G

    # Gram matrix of centered means
    gram_centered = means_centered @ means_centered.T

    try:
        # Log-determinant for numerical stability
        sign, logdet = np.linalg.slogdet(gram_centered + 1e-8 * np.eye(num_classes))
        nc5 = np.exp(logdet / num_classes) if sign > 0 else 1e-10
    except:
        nc5 = 1e-10

    return nc5

def compute_nc_metrics(model, loader, device):
    
    #Compute Neural Collapse metrics (NC1-NC5) - MAIN FUNCTION
    
    model.eval()
    features_list, labels_list = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            _ = model(xb)
            if model.last_features is not None:
                features_list.append(model.last_features.cpu())
                labels_list.append(yb)

    H = torch.cat(features_list).numpy()
    Y = torch.cat(labels_list).numpy()
    num_classes = 10

    # Compute all NC metrics
    nc1 = compute_nc1(H, Y, num_classes)
    nc2, means_norm = compute_nc2(H, Y, num_classes)
    nc3 = compute_nc3(model, means_norm, num_classes)
    nc4 = compute_nc4(H, means_norm, Y)
    nc5 = compute_nc5(H, Y, num_classes)

    # Class means for return
    class_means = np.stack([H[Y == k].mean(0) for k in range(num_classes)])

    return {
        'NC1_collapse': nc1,
        'NC2_etf': nc2,
        'NC3_duality': nc3,
        'NC4_ncc': nc4,
        'NC5_volume': nc5
    }, H, class_means

