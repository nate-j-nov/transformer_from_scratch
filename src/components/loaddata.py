# File to test loading data from pytorch.
import spacy
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import Tensor
import torch
from torch.utils.data import dataset

#https://pytorch.org/tutorials/beginner/transformer_tutorial.html
def train(): 
    EPOCHS = 0 


def loadAndBatch(): 
    train_iter = Multi30k('train'); 
    tokenizer = get_tokenizer('basic_english') # TODO: This needs to be changed, this is purely for setting up training at the moment. 
    vocab = build_vocab_from_iterator(map(tokenizer,train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor: 
        data=[torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    
    # train_iter was consumed by the process of building the vocab. So, we need to create it again. 
    train_iter, val_iter, test_iter = Multi30k(split=('train', 'valid', 'test'), language_pair = ('de', 'en'))
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

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

def main(): 
    train_iter = Multi30k(split='train', language_pair=('de', 'en'));

    tokens = []; 
    for label, line in train_iter: 
        print(f"label: {label}")
        print(f"line: {line}")
        break;

    print(tokens); 


    return 

if __name__ == "__main__": 
    main()