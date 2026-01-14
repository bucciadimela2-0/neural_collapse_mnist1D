import torch

# training/experiments
from experiments.opt_comparison import run_optimizer_comparison
from model.cnn1D import CNN1D
from model.cnn1D_layerwise import CNN1D_LayerWise
from utils.data import load_mnist1d
from utils.model_utils import train
# plotting
from utils.plot_utils import (make_img_dir, plot_geometric_analysis,
                              plot_nc_metrics, plot_optimizer_comparison,
                              plot_training_metrics,plot_layerwise_grid)


def main():
    
    # Data
    IMG_DIR, RUN_TS = make_img_dir("img")

    x_train, y_train, x_test, y_test = load_mnist1d()
    print(f"MNIST-1D Dataset: Train {x_train.shape}, Test {x_test.shape}")

    
    # Device (string, non torch.device)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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
    paths = plot_optimizer_comparison(results, out_dir=IMG_DIR, run_ts=RUN_TS)

    print("\nSaved figures:")
    for k, v in paths.items():
        if k.endswith("_fig") and v is not None:
            print(f" - {k}: {v}")
 



'''
    model = CNN1D().to(device) 
    model, history, final_metrics, H, means, tpt_epoch = train(x_train, y_train, x_test, y_test, model, device = device) 
    print(f"Test Accuracy: {history['test_acc'][-1]:.4f}") 
    print(f"NC1_collapse: {final_metrics['NC1_collapse']:.4f}") 
    print(f"NC2_etf: {final_metrics['NC2_etf']:.4f}") 
    print(f"NC3_duality: {final_metrics['NC3_duality']:.4f}") 
    print(f"NC4_ncc: {final_metrics['NC4_ncc']:.4f}")
    print(f"NC5_volume: {final_metrics['NC5_volume']:.4e}") 
   

    print(f"TPT Start Epoch: {tpt_epoch if tpt_epoch else 'Not reached'}") 
    plot_training_metrics(history, tpt_epoch, out_dir=IMG_DIR, run_ts=RUN_TS) 
    plot_nc_metrics(history, tpt_epoch, out_dir=IMG_DIR, run_ts=RUN_TS) 
    plot_geometric_analysis(model, final_metrics, H, means, y_test, out_dir=IMG_DIR, run_ts=RUN_TS) 

   layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'penultimate']
    
    # Create model
    model = CNN1D_LayerWise(num_classes=10)
    
    # Train with layer tracking
    model, history, history_layerwise, final_metrics, H, means, tpt_epoch = train(
        x_train, y_train, x_test, y_test,
        model=model,
        layer_names=layer_names,
        num_epochs=200,
        eval_every=1,
        device=device
    )

    
    print("LAYER-WISE NEURAL COLLAPSE SUMMARY")
   
    print(f"{'Layer':<15} {'NC1':<12} {'NC2':<12} {'NC4':<12}")
    print("-"*70)
    
    for layer_name in layer_names:
        nc1 = history_layerwise[layer_name]['NC1'][-1]
        nc2 = history_layerwise[layer_name]['NC2'][-1]
        nc4 = history_layerwise[layer_name]['NC4'][-1]
        print(f"{layer_name:<15} {nc1:<12.4f} {nc2:<12.4f} {nc4:<12.4f}")

    plot_layerwise_grid(history_layerwise, layer_names, out_dir=IMG_DIR, run_ts=RUN_TS)
  
    

   '''

if __name__ == "__main__":
    main()
