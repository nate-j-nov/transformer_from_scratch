# File to test loading data from pytorch.
import spacy
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import Tensor
import torch
from torch.utils.data import dataset
from typing import Tuple
import torch.nn as nn
import math
import copy

#https://pytorch.org/tutorials/beginner/transformer_tutorial.html
def train(): 
    EPOCHS = 0 

def yield_tokens(data_iter, tokenizer, index): 
    count = 0
    for from_to_tuple in data_iter: 
        count += 1
        yield tokenizer(from_to_tuple[index])

def loadAndBatch(): 
    train_iter, val_iter = Multi30k(root='.data', split=('train', 'valid'));

    tokenizer = get_tokenizer('basic_english') # TODO: This needs to be changed, this is purely for setting up training at the moment. 
    germanVocab = build_vocab_from_iterator(yield_tokens(train_iter + val_iter, tokenizer, index=0), specials=['<unk>']); 
    englishVocab = build_vocab_from_iterator(yield_tokens(train_iter + val_iter, tokenizer, index=1), specials=['<unk>'])

    englishVocab.set_default_index(englishVocab['<unk>'])
    germanVocab.set_default_index(germanVocab['<unk>'])

    def data_process(raw_text_iter: dataset.IterableDataset, vocab) -> Tensor: 
        data=[torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    
    # train_iter was consumed by the process of building the vocab. So, we need to create it again. 
    train_iter, val_iter = Multi30k(split=('train', 'valid'), language_pair = ('de', 'en'))
    train_data = data_process(train_iter, germanVocab)
    val_data = data_process(val_iter)

    device = 'cpu'

    def batchify(data: Tensor, bsz: int) -> Tensor: 
        '''
        Function to divide the data into bsz different sequences, removing elements that wouldn't fit. 
        '''

        seq_len = data.size(0) // bsz
        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(device)
    
    batch_size = 20 # TODO: Likely incorrect
    eval_batch_size = 10 # TODO: Likely incorrect
    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size) 

def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]: 
    seq_len = min(bptt, len(source) - 1 -i)
    data = source[i:i+seq_len]

def generate_square_subsequent_mask(sz: int) -> Tensor: 
    return torch.triu(torch.ones(sz, sz) * float('inf'), diagonal = 1); 

criterion = nn.CrossEntropyLoss()
learning_rate = 1.0
bptt = 35

def train(model: nn.Module, train_data: Tensor, epoch, ntokens, optimizer, scheduler) -> None: 
    model.train()
    total_loss = 0
    log_interval = 200
    src_mask = generate_square_subsequent_mask(bptt).to('cpu'); 

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)): 
        data, targets = get_batch(train_data, i)
        seq_len = data.size(0)
        if seq_len != bptt: src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters, 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0: 
            lr = scheduler.get_last_lr()[0]
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss); 
            print(f'In epoch {epoch} | Cur Loss: {cur_loss}')
            total_loss = 0

def evaluate(model: nn.Module, eval_data: Tensor, ntokens: int, optimizer, scheduler) -> float: 
    model.eval()
    total_loss = 0
    src_mask = generate_square_subsequent_mask(bptt).to('cpu')
    with torch.nograd(): 
        for i in range(0, eval_data.size(0- 1), bptt): 
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            if seq_len != bptt: 
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


def trainAndEval(model: nn.Module, train_data, eval_data, ntokens):
    epochs = 3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=.95)
    best_val_loss = float('inf');
    best_model = None

    for i in range(i + 1, epochs + 1): 
        train(model, bptt, train_data, ntokens)
        val_loss = evaluate(model, eval_data)
        val_ppl = math.exp(val_loss)
        
        if val_loss < best_val_loss: 
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        scheduler.step()

def main(): 
    train_iter = Multi30k(split='train', language_pair=('de', 'en'));
    tokens = []; 
    for label, line in train_iter: 
        print(f"label: {label}")
        print(f"line: {line}")
        break;

    print(tokens); 

if __name__ == "__main__": 
    loadAndBatch(); 