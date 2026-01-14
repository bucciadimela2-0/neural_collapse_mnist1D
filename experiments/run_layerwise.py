import os
import logging

from model.cnn1D_layerwise import CNN1D_LayerWise
from utils.model_utils import *
from utils.plot_utils import *


def run_layerwise(
    x_train, y_train, x_test, y_test,
    IMG_DIR, RUN_TS,
    num_epochs=200,
    device="cpu"
):
    
    #Experiment 2 — Layer-wise Emergence of Neural Collapse
    #Trains CNN1D_LayerWise and logs layer-wise NC metrics.
    

    layer_names = ["conv1", "conv2", "conv3", "conv4", "penultimate"]

    #Setup logging file
    log_path = os.path.join(IMG_DIR, f"{RUN_TS}_experiment2_layerwise.log")

    logger = logging.getLogger("layerwise_nc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("=" * 80)
    logger.info("EXPERIMENT 2 — LAYER-WISE EMERGENCE OF NEURAL COLLAPSE")
    logger.info("-" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Layers tracked: {layer_names}")
    logger.info(f"Output directory: {IMG_DIR}")
    logger.info("-" * 80)


    #Model + Training
    model = CNN1D_LayerWise(num_classes=10).to(device)

    model, history, history_layerwise, final_metrics, H, means, tpt_epoch = train(
        x_train, y_train, x_test, y_test,
        model=model,
        layer_names=layer_names,
        num_epochs=num_epochs,
        eval_every=1,
        device=device
    )

    
    sample_layer = layer_names[-1]  # penultimate
    keys = history_layerwise[sample_layer].keys()

    KEY_NC1 = "NC1" if "NC1" in keys else "NC1_collapse"
    KEY_NC2 = "NC2" if "NC2" in keys else "NC2_etf"
    KEY_NC4 = "NC4" if "NC4" in keys else "NC4_ncc"

    #Summary
    logger.info("LAYER-WISE NEURAL COLLAPSE SUMMARY (last logged epoch)")
    header = f"{'Layer':<15} {KEY_NC1:<12} {KEY_NC2:<12} {KEY_NC4:<12}"
    logger.info(header)
    logger.info("-" * 70)

    print("\nLAYER-WISE NEURAL COLLAPSE SUMMARY")
    print(header)
    print("-" * 70)

    for lname in layer_names:
        # Defensive: handle missing metric or empty list
        def _last(hist, k):
            return hist.get(k, [float("nan")])[-1] if len(hist.get(k, [])) else float("nan")

        nc1 = _last(history_layerwise[lname], KEY_NC1)
        nc2 = _last(history_layerwise[lname], KEY_NC2)
        nc4 = _last(history_layerwise[lname], KEY_NC4)

        row = f"{lname:<15} {nc1:<12.4f} {nc2:<12.4f} {nc4:<12.4f}"
        logger.info(row)
        print(row)

    if tpt_epoch is not None:
        logger.info(f"TPT detected at epoch: {tpt_epoch}")
    else:
        logger.info("TPT not reached")

    logger.info("Saving layer-wise grid plot...")
    path_grid = plot_layerwise_grid(history_layerwise, layer_names, out_dir=IMG_DIR, run_ts=RUN_TS)

    if path_grid is not None:
        logger.info(f"Layer-wise grid saved: {path_grid}")
    else:
        logger.info("Layer-wise grid saved (no path returned by plot_layerwise_grid).")

    logger.info(f"Log saved at: {log_path}")
    logger.info("-" * 80)

    return {
        "model": model,
        "history": history,
        "history_layerwise": history_layerwise,
        "final_metrics": final_metrics,
        "tpt_epoch": tpt_epoch,
        "log_path": log_path,
        "layerwise_fig": path_grid,
    }
