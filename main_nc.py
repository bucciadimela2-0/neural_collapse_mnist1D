import torch

# Experiment runners
from experiments.run_layerwise import run_layerwise
from experiments.run_opt_comparison import run_optimizer_comparison
from experiments.run_standard import run_standard

# Utilities
from utils.data import load_mnist1d
from utils.plot_utils import make_img_dir


def main():
    """
    Main entry point for Neural Collapse experiments.

    This script runs controlled experiments designed to study:
    - baseline Neural Collapse emergence (SGD),
    - layer-wise collapse dynamics,
    - optimizer-dependent collapse behavior.

    All results (logs + figures) are saved in a timestamped folder.
    """

    # Output directory (timestamped)
    IMG_DIR, RUN_TS = make_img_dir("img")

   
    # Load dataset
    x_train, y_train, x_test, y_test = load_mnist1d()
    print(f"MNIST-1D Dataset loaded:")
    print(f"  Train: {x_train.shape}")
    print(f"  Test : {x_test.shape}")

    # Device selection
    # Use string ('cuda' / 'cpu') for compatibility across modules
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
 
    # Optimizers for Experiment 3
    optimizer_list = ["SGD", "Adam", "LBFGS"]

    # EXPERIMENT 1 — Baseline Neural Collapse (SGD)
    run_standard(
        x_train, y_train,
        x_test, y_test,
        IMG_DIR=IMG_DIR,
        RUN_TS=RUN_TS,
        device=device
    )

    
    # EXPERIMENT 2 — Layer-wise Emergence of Neural Collapse

    # Uncomment to run
    # run_layerwise(
    #     x_train, y_train,
    #     x_test, y_test,
    #     IMG_DIR=IMG_DIR,
    #     RUN_TS=RUN_TS,
    #     device=device
    # )

   
    # EXPERIMENT 3 — Optimization Effects on Neural Collapse
   
    # Uncomment to run
    # run_optimizer_comparison(
    #     x_train, y_train,
    #     x_test, y_test,
    #     IMG_DIR=IMG_DIR,
    #     RUN_TS=RUN_TS,
    #     optimizer_list=optimizer_list,
    #     device=device
    # )


if __name__ == "__main__":
    main()
