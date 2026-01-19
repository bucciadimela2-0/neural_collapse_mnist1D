import logging
import os

from model.cnn1D import CNN1D
from utils.model_utils import *
from utils.plot_utils import *


def run_optimizer_comparison(
    x_train, y_train, x_test, y_test,
    IMG_DIR, RUN_TS,
    optimizer_list=None,
    num_epochs=200,
    device="cpu"
):
    
    #Experiment 3 — Optimization Effects on Neural Collapse
    #Trains the same CNN1D with multiple optimizers, logs results, and saves comparison figures.
    

    if optimizer_list is None:
        optimizer_list = ["SGD", "Adam"]

   
    log_path = os.path.join(IMG_DIR, f"{RUN_TS}_experiment3_opt_comparison.log")

    # Avoid duplicate handlers if function is called multiple times
    logger = logging.getLogger("opt_comparison")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("-" * 80)
    logger.info("EXPERIMENT 3 — OPTIMIZATION EFFECTS ON NEURAL COLLAPSE")
    logger.info("-" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Optimizers: {optimizer_list}")
    logger.info(f"Output directory: {IMG_DIR}")
    logger.info("-" * 80)

    results = {}
    model_fn = lambda: CNN1D(num_classes=10)

    for opt_name in optimizer_list:
        logger.info(f"\n--- Optimizer: {opt_name} ---")

        # Create fresh model
        model = model_fn().to(device)

        # Train (reuses your main train loop)
        model, history, metrics, H, means, tpt = train(
            x_train, y_train, x_test, y_test,
            model=model,
            optimizer_name=opt_name,
            num_epochs=num_epochs,
            eval_every=1,
            device=device
        )

        # Store results
        results[opt_name] = {
            "model": model,
            "history": history,
            "final_metrics": metrics,
            "H": H,
            "means": means,
            "tpt_epoch": tpt
        }

        # Log summary
        test_acc = history["test_acc"][-1] if len(history["test_acc"]) else float("nan")
        logger.info("SUMMARY")
        logger.info(f"Final Test Acc    : {test_acc:.4f}")
        logger.info(f"NC1 (Collapse)    : {metrics['NC1_collapse']:.4f}")
        logger.info(f"NC2 (ETF)         : {metrics['NC2_etf']:.4f}")
        logger.info(f"NC3 (Duality)     : {metrics['NC3_duality']:.4f}")
        logger.info(f"NC4 (NCC)         : {metrics['NC4_ncc']:.4f}")
        logger.info(f"NC5 (Volume)      : {metrics.get('NC5_volume', float('nan')):.4e}")

        if tpt is not None:
            logger.info(f"TPT Epoch         : {tpt}")
        else:
            logger.info("TPT Epoch         : N/A")

        logger.info("-" * 80)

   
    logger.info("Generating optimizer comparison plots...")
    paths = plot_optimizer_comparison(results, out_dir=IMG_DIR, run_ts=RUN_TS)

    logger.info("Saved figures:")
    for k, v in paths.items():
        if k.endswith("_fig") and v is not None:
            logger.info(f" - {k}: {v}")

    logger.info(f"Log saved at: {log_path}")
    logger.info("-" * 80)

    return results, paths, log_path
