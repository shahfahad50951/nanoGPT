# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import os

# Setup hyperparameters
batch_size = 32
seq_len = 8
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_dim = 32

# Setup fixed seed for reproduceablity
torch.manual_seed(1337)

os.system('wget \'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt\'')

with open('input.txt', 'r') as f:
    data = f.read()
    print(f'Number of characters in text dataset: {len(data)}')

# Generate vocabulary (Set of all possible tokens)
vocab = sorted(list(set(data)))
vocab_size = len(vocab)
print(f'Length of Vocabulary: {len(vocab)}')

# Tokenization (Converting raw string of text into tokens)
stoi = {ch: idx for idx,ch in enumerate(vocab)}
itos = {idx: ch for ch, idx in stoi.items()}
# Converts raw input text to tokens
encode = lambda input_str: [stoi[ch] for ch in input_str]
# Convert the integer tokens into readable strings
decode = lambda input_tokens: "".join([itos[idx] for idx in input_tokens])

print(encode('hii there'))
print(decode(encode('hii there')))
print(f'Vocabulary: {"".join(vocab)}')

# Encode tiny shakespear using the above character level tokenizer
data_t = torch.tensor(encode(data), dtype=torch.int64)
print(data_t[:1000])

# Train/valid split
n = int(0.90 * len(data_t))
train_t = data_t[:n]
valid_t = data_t[n:]
print(f'Number of characters for training: {len(train_t)}')
print(f'Number of characters for validation: {len(valid_t)}')

# Dataloader (to generate batch)
torch.manual_seed(1337)
seq_len = 8 # Determines the maximum context (input len) that the model can see to make prediction. Also called time dimension
batch_size = 4 # Deteremines the number of independent seqeunces that will be processed in one pass

def get_batch(split):
    data = train_t if split == 'train' else valid_t
    sample_idxs = torch.randint(0, len(data) - seq_len, size=(batch_size,))
    x = torch.stack([data[start_idx:start_idx+seq_len] for start_idx in sample_idxs])
    y = torch.stack([data[start_idx+1:start_idx+seq_len+1] for start_idx in sample_idxs])
    x = x.to(device)
    y = y.to(device)
    return x,y

x, y = get_batch('train') # Returens data in the dimensions: (Batch, Time) (B,T)
print(x)
print(y)

@torch.no_grad()
def estimate_losses():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for step in range(eval_iters):
            x, y = get_batch(split)
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            losses[step] = loss.item()
        out[split] = losses.mean().to('cpu')
    model.train()
    return out

class AttentionHead(nn.Module):
  def __init__(self, n_seq, n_embed, head_size, dropout=0.2):
    super().__init__()
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('mask', torch.tril(torch.ones(n_seq, n_seq)) == 0)
    self.dropout = nn.Dropout(dropout)

  def forward(self, input):
    n_batch, n_seq, n_embed = input.shape
    query = self.query(input)
    key = self.key(input)
    value = self.value(input)
    n_head = query.shape[-1]

    scores = (query @ key.transpose(-1, -2)) / (n_head ** (1/2))
    scores = scores.masked_fill(self.mask[:n_seq, :n_seq], float('-inf'))
    scores = nn.functional.softmax(scores, dim=-1)
    scores = self.dropout(scores)

    output = scores @ value
    return output

class MultiHeadAttention(nn.Module):
  def __init__(self, n_seq, n_embed, num_heads, head_size, dropout=0.2):
   super().__init__()
   assert num_heads * head_size == n_embed, 'Num Heads * Head Size should be equal to Embedding dimension size'
   self.multihead_attention = nn.ModuleList([AttentionHead(n_seq, n_embed, head_size) for _ in range(num_heads)])
   self.proj = nn.Linear(n_embed, n_embed)
   self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    output = torch.cat([self_attention(x) for self_attention in self.multihead_attention], dim=-1)
    output = self.proj(output)
    output = self.dropout(output)
    return output

class FeedForward(nn.Module):
  def __init__(self, n_embed, dropout=0.2):
    super().__init__()
    self.ffn = nn.Sequential(nn.Linear(n_embed, 4 * n_embed),
                             nn.ReLU(),
                             nn.Linear(4 * n_embed, n_embed),
                             nn.Dropout(dropout))

  def forward(self, x):
    output = self.ffn(x)
    return output

class Block(nn.Module):
  def __init__(self, n_seq, n_embed, num_heads):
    super().__init__()
    head_size = n_embed // num_heads
    self.multihead_attention = MultiHeadAttention(n_seq, n_embed, num_heads, head_size)
    self.ffn = FeedForward(n_embed)
    self.layernorm1 = nn.LayerNorm(n_embed)
    self.layernorm2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    output = x + self.multihead_attention(self.layernorm1(x))
    output = x + self.ffn(self.layernorm2(output))
    return output

class BasicGPT(nn.Module):
  def __init__(self, n_seq, num_blocks, n_embed, num_heads):
    super().__init__()
    self.n_seq, self.n_embed, self.num_heads = n_seq, n_embed, num_heads
    self.token_embedding = nn.Embedding(vocab_size, n_embed)
    self.position_embedding = nn.Embedding(n_seq, n_embed)
    self.blocks = nn.Sequential(*[Block(n_seq, n_embed, num_heads) for _ in range(num_blocks)])
    self.layernorm = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, input, target=None):
    n_batch, n_seq = input.shape
    token_embedding = self.token_embedding(input)
    position_embedding = self.position_embedding(torch.arange(n_seq, device=device))
    logits = token_embedding + position_embedding
    logits = self.blocks(logits)
    logits = self.layernorm(logits)
    logits = self.lm_head(logits)

    loss = None
    if target is not None:
      logits = logits.view(-1, vocab_size)
      target = target.view(-1)
      loss = nn.functional.cross_entropy(logits, target)

    return logits,loss

  def generate(self, input, max_new_tokens):
    self.eval()
    input_copy = input.detach().clone()
    for _ in range(max_new_tokens):
      input = input[:, -self.n_seq:]
      logits, loss = self(input)
      logits = logits[:, -1, :]
      logits = nn.functional.softmax(logits, dim=-1)
      sampled_idx = torch.multinomial(logits, 1)
      input = torch.cat((input, sampled_idx), dim=1)
      input_copy = torch.cat((input_copy, sampled_idx), dim=1)
    self.train()
    return input_copy

# Setup hyperparameters
batch_size = 64
seq_len = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_dim = 384
num_heads = 6
num_blocks = 6
dropout = 0.2
if torch.cuda.is_available(): print(f'---- Using CUDA Device ----')

# Train the model
model = BasicGPT(seq_len, num_blocks, embed_dim, num_heads)
model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_losses()
        print(f'Iteration: {step}\nTraining Loss: {losses["train"]}\nValidation Loss: {losses["val"]}')

    inputs, targets = get_batch(split='train')
    logits, loss = model(inputs, targets)
    optim.zero_grad()
    loss.backward()
    optim.step()

inputs = torch.zeros(1, 1, dtype=torch.int64, device=device)
outputs = model.generate(inputs, max_new_tokens=300)
# print(decode(outputs[0].tolist()))
for output in outputs:
  print(decode(output.tolist()))