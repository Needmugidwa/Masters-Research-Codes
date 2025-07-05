import flwr as fl
import torch
import torch.nn as nn
import timm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import time
from typing import Dict, List, Tuple, Optional
from flwr.common import Metrics, Parameters, NDArrays
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy


class FederatedMetricsTracker:
    """Track and visualize federated learning metrics."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.best_accuracy = 0.0
        self.best_round = 0

    def update(self, round_num: int, metric_type: str, value: float):
        """Update metrics storage."""
        if value is not None:
            self.metrics[f"{metric_type}_values"].append(value)
            self.metrics[f"{metric_type}_rounds"].append(round_num)

            if metric_type == "eval_accuracy" and value > self.best_accuracy:
                self.best_accuracy = value
                self.best_round = round_num

    def plot_metrics(self, filename: str = "fl_metrics.png"):
        """Generate and save metrics visualization."""
        plt.figure(figsize=(15, 10))

        # Create subplot for loss
        plt.subplot(2, 1, 1)
        if "train_loss_values" in self.metrics:
            plt.plot(
                self.metrics["train_loss_rounds"],
                self.metrics["train_loss_values"],
                "b-o",
                label="Training Loss"
            )
            plt.title("Training Loss")
            plt.xlabel('Round')
            plt.ylabel('Loss')
            plt.grid(True)


        plt.subplot(2, 1, 2)
        if "train_accuracy_values" in self.metrics:
            plt.plot(
                self.metrics["train_accuracy_rounds"],
                self.metrics["train_accuracy_values"],
                "g-o",
                label="Training Accuracy"
            )

        if "eval_accuracy_values" in self.metrics:
            plt.plot(
                self.metrics["eval_accuracy_rounds"],
                self.metrics["eval_accuracy_values"],
                "r-o",
                label="Evaluation Accuracy"
            )
            # Mark the best accuracy point
            plt.scatter(
                self.best_round, self.best_accuracy,
                c='gold', s=200, edgecolors='black',
                label=f'Best Eval: {self.best_accuracy:.2f}'
            )

        plt.title("Training vs Evaluation Accuracy")
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plot_path = os.path.join("results", filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Metrics plot saved to {plot_path}")


class SaveModelStrategy(FedAvg):
    """Custom strategy with robust model saving."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_tracker = FederatedMetricsTracker()
        self.current_params = None
        self.best_params = None
        self.save_dir = "saved_models"
        os.makedirs(self.save_dir, exist_ok=True)

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, fl.common.FitRes]], failures):
        """Aggregate training results and track metrics."""
        aggregated_params, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if results:
            # Calculate weighted averages
            train_loss = np.average(
                [r.metrics.get('train_loss', 0) for _, r in results],
                weights=[r.num_examples for _, r in results]
            )
            train_acc = np.average(
                [r.metrics.get('train_accuracy', 0) for _, r in results],
                weights=[r.num_examples for _, r in results]
            )

            self.metrics_tracker.update(server_round, "train_loss", train_loss)
            self.metrics_tracker.update(server_round, "train_accuracy", train_acc)
            print(f"Round {server_round} - Avg Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        if aggregated_params:
            self.current_params = aggregated_params

        return aggregated_params, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, fl.common.EvaluateRes]], failures):
        """Aggregate evaluation results and save models."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        if results:
            eval_acc = np.average(
                [r.metrics.get('accuracy', 0) for _, r in results],
                weights=[r.num_examples for _, r in results]
            )
            self.metrics_tracker.update(server_round, "eval_accuracy", eval_acc)
            print(f"Round {server_round} - Eval Accuracy: {eval_acc:.4f}")

            # Always save the current model
            self._save_model(server_round, eval_acc, is_best=False)

            # Save as best if improved
            print(f"Current accuracy: {eval_acc:.4f}, Best accuracy: {self.metrics_tracker.best_accuracy:.4f}")
            if eval_acc > self.metrics_tracker.best_accuracy:
                print("Accuracy improved - saving best model...")
            #if eval_acc > self.metrics_tracker.best_accuracy:
                print(f"New best accuracy {eval_acc:.4f} at round {server_round}")
                self._save_model(server_round, eval_acc, is_best=True)
                self.best_params = self.current_params

        return aggregated_loss, aggregated_metrics

    def _save_model(self, round_num: int, accuracy: float, is_best: bool = False):
        """Robust model saving with error handling."""
        if not self.current_params:
            print("Warning: No parameters available to save!")
            return

        try:
            # Convert parameters to model state dict
            model = load_model()
            param_arrays: List[NDArrays] = fl.common.parameters_to_ndarrays(self.current_params)
            state_dict = dict(zip(model.state_dict().keys(), [torch.from_numpy(arr) for arr in param_arrays]))

            # Verify parameter shapes match
            for name, param in model.state_dict().items():
                if state_dict[name].shape != param.shape:
                    raise ValueError(f"Shape mismatch for parameter {name}")

            model.load_state_dict(state_dict)

            # Prepare save data
            timestamp = int(time.time())
            save_data = {
                'round': round_num,
                'accuracy': accuracy,
                'model_state_dict': model.state_dict(),
                'timestamp': timestamp,
                'is_best': is_best
            }

            # Save model
            filename = f"model_round_{round_num}_acc_{accuracy:.4f}_{timestamp}.pth"
            model_path = os.path.join(self.save_dir, filename)
            torch.save(save_data, model_path)
            print(f"Model successfully saved to {model_path}")

            # Save best model separately
            if is_best:
                best_path = os.path.join(self.save_dir, "BEST_MODEL.pth")
                torch.save(model.state_dict(), best_path)
                print(f"Best model saved to {best_path}")

        except Exception as e:
            print(f"Error saving model: {str(e)}")
            import traceback
            traceback.print_exc()


def load_model():
    """Load and configure the model."""
    model = timm.create_model('cait_xxs24_224', pretrained=True)
    model.reset_classifier(5)  # For 5 classes
    return model 

'''
def load_model():
    model = timm.create_model('deit_tiny_patch16_224', pretrained=True)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze specific parts (safer approach)
    for name, param in model.named_parameters():
        # Unfreeze the classifier head
        if 'head' in name or 'fc' in name:  # Different DeiT variants use different names
            param.requires_grad = True
            continue

        # Unfreeze last N blocks (safer parsing)
        if 'blocks' in name:
            try:
                # Handle both formats: 'blocks.11.attn.qkv.weight' and 'blocks.11.norm1.weight'
                parts = name.split('.')
                block_idx = int(parts[1])  # Get the block number (e.g., 11 from 'blocks.11...')
                if block_idx >= len(model.blocks) - 2:  # Unfreeze last 2 blocks
                    param.requires_grad = True
            except (IndexError, ValueError):
                # Skip if name doesn't match expected pattern
                continue

    return model
    '''
'''
def load_model():
    model = timm.create_model('cait_xxs24_224', pretrained=True, drop_rate=0.1, attn_drop_rate=0.1)
    num_classes = 5

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last few blocks and head
    for name, param in model.named_parameters():
        if 'blocks.22' in name or 'blocks.23' in name or 'head' in name:  # Last two blocks
            param.requires_grad = True

    # Modify the head
    if hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        model.reset_classifier(num_classes)

    return model
'''
def get_initial_parameters() -> Parameters:
    """Return initial parameters for federated learning."""
    model = load_model()
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return fl.common.ndarrays_to_parameters(ndarrays)

def main():
    """Run federated learning server with robust model saving."""
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=get_initial_parameters(),

        on_fit_config_fn=lambda round_num: {
            "batch_size": 16,
            #"local_epochs": 2 + round_num // 2,
            "local_epochs":2,
            "current_round": round_num
        },
    )

    try:
        fl.server.start_server(
            server_address="0.0.0.0:8082",
            config=fl.server.ServerConfig(num_rounds=20),
            strategy=strategy,
            grpc_max_message_length=1024 * 1024 * 1024  # 1GB max message size
        )
    finally:
        # Save final metrics and best model.
        timestamp = int(time.time())
        plot_filename = f"fl_metrics_{timestamp}.png"
        strategy.metrics_tracker.plot_metrics(plot_filename)
        print(
            f"\nBest accuracy: {strategy.metrics_tracker.best_accuracy:.4f} at round {strategy.metrics_tracker.best_round}")



if __name__ == "__main__":
    main()