import torch
import math
from torch.utils.data import DataLoader
from text8_dataset import Text8
from model import GPT
import tqdm
from datasets import load_dataset
from tokenizer import Tokenizer

max_seq_len = 128
num_heads = 4 
num_blocks = 3 
batch_size = 256
hidden = 128
use_case = "test"

def generate(model, tokenizer, prompt, device):
    seq = tokenizer.encode(prompt) # => T
    seq = seq.unsqueeze(0) # => 1 x T0 
    seq = seq.to(device)
    text_res = prompt
    for i in range(200):
        seq = seq[:, -max_seq_len:]
        out = model(seq) # => 1 x Ti x vocab_size
        out = out[0,-1,:] # => vocab_size
        out = torch.multinomial(torch.exp(out), 1)
        # out = torch.argmax(out) # => 1
        out_letter = tokenizer.decode([int(out.detach())])
        text_res += out_letter
        seq = torch.concat([seq, out.unsqueeze(0)], dim=1)
    print(text_res)
    return 0


tokenizer = Tokenizer(load_dataset("afmck/text8")[use_case][0]["text"])

loss_fn = torch.nn.NLLLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dataset = Text8(max_seq_len, tokenizer, usecase=use_case)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
gpt = GPT(hidden, vocab_size=dataset.vocab_size, num_heads=num_heads, num_blocks=num_blocks, max_seq_len=max_seq_len, device=device).to(device)

optimizer = torch.optim.Adam(gpt.parameters())

for epoch in range(200):
    losses = []
    i = 0
    for x, y in tqdm.tqdm(dataloader, total=math.ceil(len(dataset)/batch_size)):
        x, y = x.to(device), y.to(device)
        B, T = x.shape
        optimizer.zero_grad()
        output = gpt(x)
        # output: B x T x vocab_size ; y: B x T 
        loss = loss_fn(output.view(B*T, -1), y.view(B*T)) # devide by batch size? 
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if i > 2000:
            print(torch.tensor(losses).mean())
            losses = []
            i = 0
            with torch.no_grad():
                gpt.eval()
                generate(gpt, tokenizer, "the world is ", device)
                gpt.train()
        i += 1
        torch.save(gpt.state_dict(), "trained_models/gpt.pt")
        



print("")
