import flwr as fl
import torch
import timm
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Parameters, FitIns, EvaluateIns, NDArrays
from flwr.server.client_proxy import ClientProxy


# Define the model architecture for initialization and evaluation
def load_model():
    model_name = 'cait_s24_224'
    model = timm.create_model(model_name, pretrained=True)

    # Modify for 5 classes
    num_classes = 5
    if hasattr(model, 'head'):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        in_features = model.get_classifier().in_features
        model.reset_classifier(num_classes)

    return model


# Define federated averaging strategy with model saving
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    # Extract accuracy values and client dataset sizes
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate accuracy
    return {"accuracy": sum(accuracies) / sum(examples)}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        # Extract and remove the initial_parameters_ndarrays from kwargs before passing to parent
        initial_params = None
        if "initial_parameters_ndarrays" in kwargs:
            initial_params = kwargs.pop("initial_parameters_ndarrays")

        super().__init__(*args, **kwargs)
        self.best_accuracy = 0.0

        # Make sure we have a list of ndarrays
        if initial_params is not None:
            if isinstance(initial_params, list):
                self.current_parameters_ndarrays = initial_params
            elif hasattr(initial_params, 'tensors'):
                # Convert Parameters object to list of NumPy arrays
                from flwr.common.parameter import bytes_to_ndarray
                self.current_parameters_ndarrays = [
                    bytes_to_ndarray(tensor) for tensor in initial_params.tensors
                ]
            else:
                self.current_parameters_ndarrays = None
        else:
            self.current_parameters_ndarrays = None

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
            failures: List[BaseException],
    ) -> Optional[float]:
        if not results:
            return None

        # Call aggregate_evaluate from base class
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)

        # Handle the aggregated result
        if aggregated_result is not None:
            # Extract accuracy from client results if available
            accuracies = []
            examples = []
            for client_proxy, evaluate_res in results:
                if evaluate_res.metrics and "accuracy" in evaluate_res.metrics:
                    accuracies.append(evaluate_res.num_examples * evaluate_res.metrics["accuracy"])
                    examples.append(evaluate_res.num_examples)

            # Compute weighted average accuracy
            if accuracies and sum(examples) > 0:
                accuracy = sum(accuracies) / sum(examples)
                print(f"Round {server_round} accuracy: {accuracy}")

                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    # Use the current parameters (latest aggregated model)
                    if self.current_parameters_ndarrays is not None:
                        # Make sure we have a list of NumPy arrays
                        params_list = self.current_parameters_ndarrays
                        if hasattr(params_list, 'tensors'):
                            # Convert Parameters object to list of NumPy arrays
                            from flwr.common.parameter import bytes_to_ndarray
                            params_list = [
                                bytes_to_ndarray(tensor) for tensor in params_list.tensors
                            ]

                        # Create a model and update it with the parameters
                        model = load_model()

                        # Convert numpy arrays to PyTorch tensors and create state dict
                        state_dict = {}
                        model_state_keys = list(model.state_dict().keys())

                        for i, param in enumerate(params_list):
                            if i < len(model_state_keys):
                                key = model_state_keys[i]
                                state_dict[key] = torch.from_numpy(param.copy())  # Make a copy to ensure memory safety

                        # Load state dict into model
                        model.load_state_dict(state_dict, strict=True)

                        # Save the model
                        torch.save(model.state_dict(), f"fl_cait_model_round_{server_round}.pth")
                        print(f"Model saved at round {server_round} with accuracy: {accuracy}")

        return aggregated_result


# Define metrics aggregation functions
def fit_metrics_aggregation_fn(metrics_list):
    """Aggregate training metrics from multiple clients."""
    # metrics_list is a list of tuples (num_examples, metrics_dict)
    if not metrics_list:
        return {}

    metrics_sum = {}
    examples_sum = 0

    # Sum up weighted metrics from all clients
    for num_examples, metrics in metrics_list:
        examples_sum += num_examples
        for key, value in metrics.items():
            if key not in metrics_sum:
                metrics_sum[key] = 0
            metrics_sum[key] += value * num_examples

    # Compute weighted averages
    return {key: value / examples_sum for key, value in metrics_sum.items()}


def evaluate_metrics_aggregation_fn(metrics_list):
    """Aggregate evaluation metrics from multiple clients."""
    if not metrics_list:
        return {}

    accuracy_sum = 0
    examples_sum = 0

    # Sum up weighted accuracies from all clients
    for num_examples, metrics in metrics_list:
        if "accuracy" in metrics:
            accuracy_sum += metrics["accuracy"] * num_examples
            examples_sum += num_examples

    # Return weighted average accuracy
    if examples_sum == 0:
        return {}
    return {"accuracy": accuracy_sum / examples_sum}


# Main server execution code
def main():
    # Initialize model
    model = load_model()

    # Get model parameters for initialization as numpy arrays
    params_ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Define strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,  # Use 100% of available clients for training
        fraction_evaluate=1.0,  # Use 100% of available clients for evaluation
        min_fit_clients=1,  # Minimum number of clients to train in each round
        min_evaluate_clients=1,  # Minimum number of clients to evaluate in each round
        min_available_clients=1,  # Minimum number of clients to start a round
        evaluate_fn=None,  # No centralized evaluation
        on_fit_config_fn=lambda round_num: {"batch_size": 2, "local_epochs": 2 + round_num // 2},
        # Increase epochs over time
        initial_parameters=fl.common.ndarrays_to_parameters(params_ndarrays),  # Initialize with pre-trained model
        #initial_parameters_ndarrays=params_ndarrays,  # Store the numpy arrays directly
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    # Set the initial parameters directly
    strategy.current_parameters_ndarrays = params_ndarrays  # This should be a list of numpy arrays
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy
    )


if __name__ == "__main__":
    main()

