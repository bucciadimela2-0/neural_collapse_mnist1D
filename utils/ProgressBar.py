import sys
from datetime import datetime, timedelta
from IPython.display import clear_output

class ProgressBar:
    def __init__(self, num_epochs, width=50):
        self.num_epochs = num_epochs
        self.width = width
        self.start_time = datetime.now()
        self.epoch_times = []  # Track individual epoch times for accurate ETA

    def update(self, epoch, train_acc, test_acc, train_loss, val_loss, nc1=None, nc2=None, nc3=None, nc4=None):
        
        clear_output(wait=True)
        
        current_time = datetime.now()
        elapsed_total = current_time - self.start_time
        progress_pct = (epoch + 1) / self.num_epochs
        
        # ETA calculation using per-epoch timing
        epoch_time = elapsed_total.total_seconds()
        self.epoch_times.append(epoch_time)
        
        if len(self.epoch_times) > 3:  # Use last 3 epochs for smooth ETA
            avg_epoch_time = (self.epoch_times[-1] - self.epoch_times[-4]) / 3
            remaining_epochs = self.num_epochs - (epoch + 1)
            eta_seconds = avg_epoch_time * remaining_epochs
            eta = timedelta(seconds=max(0, eta_seconds))
        else:
            eta = timedelta(seconds=0)
        
        # Progress bar
        filled = int(progress_pct * self.width)
        bar = "█" * filled + "░" * (self.width - filled)
        
        # NC metrics
        nc_metrics = []
        if nc1 is not None: nc_metrics.append(f"NC1:{nc1:.3f}")
        if nc2 is not None: nc_metrics.append(f"NC2:{nc2:.3f}")
        if nc3 is not None: nc_metrics.append(f"NC3:{nc3:.3f}")
        if nc4 is not None: nc_metrics.append(f"NC4:{nc4:.3f}")
        nc_str = " | ".join(nc_metrics) if nc_metrics else "NC: -"
        
        # time formatting
        elapsed_mins = int(elapsed_total.total_seconds() // 60)
        elapsed_secs = int(elapsed_total.total_seconds() % 60)
        elapsed_str = f"{elapsed_mins:02d}:{elapsed_secs:02d}"
        
        eta_mins = eta.seconds // 60
        eta_secs = eta.seconds % 60
        eta_str = f"{eta_mins:02d}:{eta_secs:02d}"
        
        table = f"""
+--------------------------------------------------------------------------------+
|[{bar:<50}] {int(progress_pct*100):3d}% |
+--------------------------------------------------------------------------------+
| Epoch {epoch+1:3d}/{self.num_epochs:3d} | Acc: {train_acc:6.4f} / {test_acc:6.4f} | Loss: {train_loss:7.4f} / {val_loss:7.4f} |
| Time: {elapsed_str:<5} ETA: {eta_str:<8} | {nc_str:<32} |
+--------------------------------------------------------------------------------+
        """.strip()
        
        print(table)
        sys.stdout.flush()
