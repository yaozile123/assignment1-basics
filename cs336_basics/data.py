import torch
import numpy as np
import numpy.typing as npt

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_size = len(dataset)
    # Generate random start indices as numpy array for faster indexing
    start_indices = np.random.randint(0, dataset_size - context_length, size=batch_size)
    
    # Use numpy advanced indexing to get all sequences at once
    input_sequences = np.stack([dataset[i:i + context_length] for i in start_indices])
    target_sequences = np.stack([dataset[i + 1:i + context_length + 1] for i in start_indices])
    
    # Convert to torch tensors only once
    sampled_inputs = torch.from_numpy(input_sequences.copy()).long()
    next_tokens = torch.from_numpy(target_sequences.copy()).long()
    
    return sampled_inputs.to(device), next_tokens.to(device)

class DataLoader:
    def __init__(self, dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return data_loading(self.dataset, self.batch_size, self.context_length, self.device)