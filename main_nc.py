import torch

# training/experiments
from experiments.opt_comparison import run_optimizer_comparison
from model.cnn1D import CNN1D
from utils.data import load_mnist1d
from utils.model_utils import train
# plotting
from utils.plot_utils import (make_img_dir, plot_geometric_analysis,
                              plot_nc_metrics, plot_optimizer_comparison,
                              plot_training_metrics)


def main():
    
    # Data
   
    x_train, y_train, x_test, y_test = load_mnist1d()
    print(f"MNIST-1D Dataset: Train {x_train.shape}, Test {x_test.shape}")

    
    # Device (string, non torch.device)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = CNN1D().to(device) 
    model, history, final_metrics, H, means, tpt_epoch = train(x_train, y_train, x_test, y_test, model, device = device) 
    print(f"Test Accuracy: {history['test_acc'][-1]:.4f}") 
    print(f"NC1_collapse: {final_metrics['NC1_collapse']:.4f}") 
    print(f"NC2_etf: {final_metrics['NC2_etf']:.4f}") 
    print(f"NC3_duality: {final_metrics['NC3_duality']:.4f}") 
    print(f"NC4_ncc: {final_metrics['NC4_ncc']:.4f}")
    print(f"NC5_volume: {final_metrics['NC5_volume']:.4e}") 
    IMG_DIR, RUN_TS = make_img_dir("img")

    print(f"TPT Start Epoch: {tpt_epoch if tpt_epoch else 'Not reached'}") 
    plot_training_metrics(history, tpt_epoch, out_dir=IMG_DIR, run_ts=RUN_TS) 
    plot_nc_metrics(history, tpt_epoch, out_dir=IMG_DIR, run_ts=RUN_TS) 
    plot_geometric_analysis(model, final_metrics, H, means, y_test, out_dir=IMG_DIR, run_ts=RUN_TS) 

    '''
    # Model factory 
    
    model_fn = lambda: CNN1D(num_classes=10)

   
    # Run experiments
  
    results = run_optimizer_comparison(
        x_train, y_train, x_test, y_test,
        model_fn=model_fn,
        optimizer_list=["SGD", "Adam", "LBFGS"],
        num_epochs=50,
        device=device,
    )

    
    # Plots (saved to img/<timestamp>/)
    
    IMG_DIR, RUN_TS = make_img_dir("img")
    #paths = plot_optimizer_comparison(results, out_dir=IMG_DIR, run_ts=RUN_TS)

    print("\nSaved figures:")
    for k, v in paths.items():
        if k.endswith("_fig") and v is not None:
            print(f" - {k}: {v}")

   '''

if __name__ == "__main__":
    main()
