
class History:
    def __init__(self, train_loss, valid_metrics, checkpoint_path):
        self.model = None
        self.checkpoint_dir = None
        self.training_log = []

class EpochHistory:
    def __init__(self, loss, metrics):
        self.loss = {loss.__class__.name:[]}
        self.metrics = {m.__class__.name:[] for m in metrics}
    
    def update(self, loss=None, metrics=None):
        if loss:
            for name, value in loss.items():
                self.loss[name].append(value)
        if metrics:
            for name, value in metrics.items():
                self.metrics[name].append(value)
    
    def __repr__(self) -> str:
        avg_train_loss = {key:value.mean() for key, value in self.loss.items()}
        avg_valid_metrics = {key:value.mean() for key, value in self.metrics.items()}
        out = ""
        for key, value in (avg_train_loss | avg_valid_metrics).items():
            out += f"Avg. {key}: {value:.3f} "
        return out