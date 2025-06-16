import torch
import numpy.typing as npt

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_size = len(dataset)
    start_indices = torch.randint(0, dataset_size - context_length, (batch_size,))
    sampled_inputs = torch.stack([torch.from_numpy(dataset[i:i + context_length]) for i in start_indices])
    next_tokens = torch.stack([torch.from_numpy(dataset[i + 1:i + context_length + 1]) for i in start_indices])
    return sampled_inputs.to(device), next_tokens.to(device)

class DataLoader:
    def __init__(self, dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return data_loading(self.dataset, self.batch_size, self.context_length, self.device)