import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.layers import softmax

def top_p_sampling(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True) # (batch_size, vocab_size)``
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1) # (batch_size, vocab_size)
    mask = cumsum_probs <= top_p
    mask[:, 0] = True
    return probs * mask

def generate_text(model: torch.nn.Module, tokenizer: Tokenizer, prompt: str, max_new_tokens: int, temperature: float = 0.8, top_p: float = 0.9) -> str:
    model.eval()
    prompt_tokens = tokenizer.encode(prompt)
    context = torch.tensor(prompt_tokens, dtype=torch.long, device=model.device).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :] / temperature
        probs = softmax(logits, dim=-1) 
        probs = top_p_sampling(probs, top_p)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=-1)
        if next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(context[0].tolist())