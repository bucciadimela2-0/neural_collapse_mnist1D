from utils.model_utils import *


def run_optimizer_comparison(x_train, y_train, x_test, y_test, 
                             model_fn, optimizer_list=None, 
                             num_epochs=300, device='cpu'):
    """
    Run experiments with multiple optimizers
    
    Args:
        model_fn: Function that returns a fresh model instance
        optimizer_list: List of optimizer names to test
    
    Returns:
        results: Dict with results for each optimizer
    """
    
    if optimizer_list is None:
        optimizer_list = ['SGD', 'Adam', 'LBFGS']
    
    results = {}
    
    for opt_name in optimizer_list:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {opt_name}")
        print(f"{'='*70}\n")
        
        # Create fresh model
        model = model_fn()
        
        # Train
        model, history, metrics, H, means, tpt = train(
            x_train, y_train, x_test, y_test,
            model=model,
            optimizer_name=opt_name,
            num_epochs=num_epochs,
            eval_every=5,  # Evaluate every 5 epochs for speed
            device=device
        )
        
        # Store results
        results[opt_name] = {
            'model': model,
            'history': history,
            'final_metrics': metrics,
            'H': H,
            'means': means,
            'tpt_epoch': tpt
        }
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"SUMMARY: {opt_name}")
        print(f"{'='*70}")
        print(f"Final Test Acc:  {history['test_acc'][-1]:.4f}")
        print(f"NC1 (Collapse):  {metrics['NC1_collapse']:.4f}")
        print(f"NC2 (ETF):       {metrics['NC2_etf']:.4f}")
        print(f"NC3 (Duality):   {metrics['NC3_duality']:.4f}")
        print(f"NC4 (NCC):       {metrics['NC4_ncc']:.4f}")
        print(f"TPT Epoch:       {tpt if tpt else 'N/A'}")
        print(f"{'='*70}\n")
    
    
    
    return results