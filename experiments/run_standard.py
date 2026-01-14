import logging
import os

from model.cnn1D import CNN1D
from utils.model_utils import *
from utils.plot_utils import *


def run_standard(
    x_train, y_train, x_test, y_test,
    IMG_DIR, RUN_TS,
    num_epochs=200,
    device='cpu'
):
    
    #Experiment 1 — Baseline Neural Collapse with SGD
    #Logs results to file and saves figures.

    log_path = os.path.join(IMG_DIR, f"{RUN_TS}_experiment1.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    logger.info("-" * 70)
    logger.info("EXPERIMENT 1 — BASELINE NEURAL COLLAPSE WITH SGD")
    logger.info("-" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Output directory: {IMG_DIR}")
    logger.info("-" * 70)

   
    model = CNN1D().to(device)

    logger.info("Model initialized: CNN1D")

    model, history, final_metrics, H, means, tpt_epoch = train(
        x_train, y_train, x_test, y_test,
        model,
        num_epochs=num_epochs,
        device=device
    )

  
    logger.info("FINAL RESULTS")
    logger.info(f"Test Accuracy     : {history['test_acc'][-1]:.4f}")
    logger.info(f"NC1 (Collapse)    : {final_metrics['NC1_collapse']:.4f}")
    logger.info(f"NC2 (ETF)         : {final_metrics['NC2_etf']:.4f}")
    logger.info(f"NC3 (Duality)     : {final_metrics['NC3_duality']:.4f}")
    logger.info(f"NC4 (NCC)         : {final_metrics['NC4_ncc']:.4f}")
    logger.info(f"NC5 (Volume)      : {final_metrics['NC5_volume']:.4e}")

    if tpt_epoch is not None:
        logger.info(f"TPT detected at epoch: {tpt_epoch}")
    else:
        logger.info("TPT not reached")

   
    logger.info("Saving plots...")

    plot_training_metrics(
        history, tpt_epoch,
        out_dir=IMG_DIR, run_ts=RUN_TS
    )

    plot_nc_metrics(
        history, tpt_epoch,
        out_dir=IMG_DIR, run_ts=RUN_TS
    )

    plot_geometric_analysis(
        model, final_metrics, H, means, y_test,
        out_dir=IMG_DIR, run_ts=RUN_TS
    )

    logger.info("Experiment completed successfully.")
    logger.info("-" * 70)

    return {
        "model": model,
        "history": history,
        "metrics": final_metrics,
        "tpt_epoch": tpt_epoch,
        "log_path": log_path,
    }
