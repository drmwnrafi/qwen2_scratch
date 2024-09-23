import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
import tiktoken
from collections import dataclass
from model import Qwen2

@dataclass

def tokenize_dummy_sentences(sentences, tokenizer):
    tokens = [tokenizer.encode(sentence) for sentence in sentences]
    max_length = max(len(t) for t in tokens)
    token_ids = [t + [tokenizer.eot_token] * (max_length - len(t)) for t in tokens]
    return torch.tensor(token_ids, dtype=torch.long), max_length

class DummyDataset(Dataset):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.token_ids[idx]

tokenizer = tiktoken.get_encoding("gpt2")

class ModelConfig:
    vocab_size: int = tokenizer.n_vocab
    embedding_size: int = 512
    n_layers: int = 12
    n_q_heads: int = 8
    n_kv_heads: int = 2
    intermediate_dim: int = 4864
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-8
    swiglu_beta: float = 1.0
    bias: bool = False

dummy_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I love machine learning and deep learning techniques.",
    "Artificial Intelligence is fascinating.",
    "Language models are a crucial part of AI research.",
    "This is an example of a dummy sentence.",
    "I am learning how to build transformers.",
    "Neural networks can be powerful tools.",
    "Backpropagation helps models learn.",
    "Reinforcement learning is an exciting area.",
    "Optimization algorithms improve model performance.",
    "Convolutional neural networks are used in vision tasks.",
    "NLP models need large datasets.",
    "Attention mechanisms improve model accuracy.",
    "Training models can take time and resources.",
    "Fine-tuning pre-trained models is common.",
    "GPT models generate human-like text.",
    "AI is transforming various industries.",
    "I am experimenting with transformers.",
    "PyTorch is a popular deep learning framework.",
    "Training neural networks involves a lot of computation."
]

token_ids, seq_len = tokenize_dummy_sentences(dummy_sentences, tokenizer)
dataset = DummyDataset(token_ids)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


config = ModelConfig()
model = Qwen2(config=ModelConfig)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
mask = torch.tril(torch.ones(1, seq_len, seq_len)).to(device)

n_epochs = 20
model.train()
for epoch in range(n_epochs):
    total_loss = 0
    for tokens, targets in dataloader:
        tokens, targets = tokens.to(device), targets.to(device)
        logits = model(tokens, mask)
        optimizer.zero_grad()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()

        loss = criterion(shift_logits.view(-1, tokenizer.n_vocab), shift_targets.view(-1))
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")

print("Training finished!")

model.eval()
with torch.no_grad():
    input_text = "GPT"
    input_tokens = tokenizer.encode(input_text)
    input_tokens = torch.tensor([input_tokens], dtype=torch.long).to(device)

    generated_tokens = input_tokens
    max_length = 20
    for _ in range(max_length - len(input_tokens[0])):
        logits = model(generated_tokens)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)

    generated_text = tokenizer.decode(generated_tokens[0].cpu().numpy().tolist())
    print(f"Generated text: {generated_text}")