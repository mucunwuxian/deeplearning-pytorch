import os
import numpy as np
import torch
import time
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from datasets import load_small_parallel_enja
from utils import pad_sequences, sort_sequences
from sklearn.utils import shuffle


class Encoder(nn.Module):
    def __init__(self, vocab, hidden_dim=256, device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        self.embedding = nn.Embedding(vocab, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)

    def forward(self, x, h):
        x = self.embedding(x).view(1, 1, -1)
        x, h = self.gru(x, h)
        return x, h

    def zero_state(self):
        return torch.zeros(1, 1, self.hidden_dim, device=self.device)


class Decoder(nn.Module):
    def __init__(self, vocab, hidden_dim=256,
                 p_dropout=0.1, maxlen=30, device='cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.device = device

        self.embedding = nn.Embedding(vocab, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, maxlen)
        self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(p_dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab)

    def forward(self, x, h, encoder_outputs):
        x = self.embedding(x).view(1, 1, -1)
        x = self.dropout(x)

        attn_weights = F.softmax(
            self.attn(torch.cat((x[0], h[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        x = torch.cat((x[0], attn_applied[0]), 1)
        x = self.attn_combine(x).unsqueeze(0)

        x = F.relu(x)
        x, h = self.gru(x, h)
        x = F.log_softmax(self.out(x[0]), dim=1)

        return x, h, attn_weights

    def zero_state(self):
        return torch.zeros(1, 1, self.hidden_dim, device=self.device)


def train_iters(encoder, decoder, src, tgt, n_iters,
                start_char=0, end_char=1,
                maxlen=30,
                print_every=1000, device='cpu'):
    assert len(src) == len(tgt)

    start = time.time()
    losses = []
    print_loss_total = 0

    encoder_optimizer = torch.optim.Adam(encoder.parameters())
    decoder_optimizer = torch.optim.Adam(decoder.parameters())
    indices = shuffle(np.arange(len(src), dtype=np.int32))
    pairs = [to_tensor_pairs(i, src, tgt, device) for i in indices]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters+1):
        pair = pairs[iter-1]
        source = pair[0]
        target = pair[1]

        loss = train(encoder, decoder, source, target,
                     encoder_optimizer, decoder_optimizer,
                     criterion, start_char=start_char, end_char=end_char,
                     maxlen=maxlen,
                     device=device)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('iter: {}, loss: {:.3}'.format(iter, print_loss_avg))


def train(encoder, decoder,
          source_tensor, target_tensor,
          encoder_optimizer, decoder_optimizer,
          criterion, maxlen=30,
          start_char=0, end_char=1,
          teacher_forcing_ratio=0.7,
          device='cpu'):
    encoder_hidden = encoder.zero_state()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    source_length = source_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(maxlen, encoder.hidden_dim, device=device)

    loss = 0

    for t in range(source_length):
        encoder_output, encoder_hidden = \
            encoder(source_tensor[t], encoder_hidden)
        encoder_outputs[t] = encoder_output[0, 0]

    decoder_input = torch.tensor([[start_char]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True \
        if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(target_length):
            decoder_output, decoder_hidden, decoder_attn = \
                decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[t])
            decoder_input = target_tensor[t]
    else:
        for t in range(target_length):
            decoder_output, decoder_hidden, decoder_attn = \
                decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[t])
            if decoder_input.item() == end_char:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def evaluate_iters(encoder, decoder, src, tgt,
                   src_i2w, tgt_i2w,
                   n=10,
                   start_char=0, end_char=1,
                   maxlen=30,
                   device='cpu'):
    assert len(src) == len(tgt)

    def decode(sentence, i2w):
        return ' '.join([i2w[i] for i in sentence])

    indices = shuffle(np.arange(len(src), dtype=np.int32))
    for i in range(n):
        source = src[i]
        target = tgt[i]

        outputs, attns = evaluate(encoder, decoder, source,
                                  maxlen, start_char, end_char, device)

        print('>', decode(source[1:-1], src_i2w))
        print('=', decode(target[1:-1], tgt_i2w))
        print('<', decode(outputs, tgt_i2w))
        print()


def evaluate(encoder, decoder, sentence,
             maxlen=30, start_char=0, end_char=0,
             device='cpu'):
    with torch.no_grad():
        source_tensor = to_tensor(sentence, device=device)
        source_length = source_tensor.size(0)
        encoder_hidden = encoder.zero_state()
        encoder_outputs = \
            torch.zeros(maxlen, encoder.hidden_dim, device=device)

        for t in range(source_length):
            encoder_output, encoder_hidden = \
                encoder(source_tensor[t], encoder_hidden)
            encoder_outputs[t] = encoder_output[0, 0]

        decoder_input = torch.tensor([[start_char]], device=device)
        decoder_hidden = encoder_hidden

        decoder_words = []
        decoder_attns = torch.zeros(maxlen, maxlen, device=device)

        for t in range(maxlen):
            decoder_output, decoder_hidden, decoder_attn = \
                decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attns[t] = decoder_attn.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == end_char:
                decoder_words.append(end_char)
                break
            else:
                decoder_words.append(topi.item())
            decoder_input = topi.squeeze().detach()

        return decoder_words, decoder_attns[:t+1]


def to_tensor_pairs(idx, src, tgt, device='cpu'):
    src = to_tensor(src[idx], device=device)
    tgt = to_tensor(tgt[idx], device=device)
    return (src, tgt)


def to_tensor(sentence, device='cpu'):
    return torch.tensor(sentence[1:], device=device).view(-1, 1)


if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    Load data
    '''
    pad_value = -1
    start_char = 0
    end_char = 1
    oov_char = 2
    index_from = 3
    (train_x, train_y), (test_x, test_y), (num_x, num_y), \
        (w2i_X, w2i_y), (i2w_x, i2w_y) = \
        load_small_parallel_enja(to_ja=True,
                                 pad_value=pad_value,
                                 start_char=start_char,
                                 end_char=end_char,
                                 oov_char=oov_char,
                                 index_from=index_from)

    train_x, train_y = sort_sequences(train_x, train_y)
    test_x, test_y = sort_sequences(test_x, test_y)

    train_size = 50000  # up to 50000
    test_size = 500     # up to 500
    train_x, train_y = train_x[:train_size], train_y[:train_size]
    test_x, test_y = test_x[:test_size], test_y[:test_size]

    '''
    Build model
    '''
    maxlen = 30
    encoder = Encoder(num_x, device=device)
    decoder = Decoder(num_y, device=device, maxlen=maxlen)

    '''
    Train model
    '''
    epochs = 10
    n_iters = len(train_x) // 2
    print_every = n_iters // 5
    for epoch in range(epochs):
        print('epoch: {}'.format(epoch+1))
        train_iters(encoder, decoder, train_x, train_y, n_iters,
                    start_char=start_char, end_char=end_char,
                    maxlen=maxlen,
                    print_every=print_every, device=device)
        evaluate_iters(encoder, decoder, test_x, test_y, i2w_x, i2w_y,
                       n=10, start_char=start_char, end_char=end_char,
                       maxlen=maxlen, device=device)
