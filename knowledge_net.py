import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import itertools
import data
import knowledge_base as kb

torch.manual_seed(50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# todo:
# device, check backpropagating, batches, valid vs test, real optimizer


class KnowledgeRNN(nn.Module):
    def __init__(self,
                 ntokens,
                 state_size,
                 input_embed_size,
                 knowledge_base=None,
                 value_size=80,
                 query_size=40,
                 dropout=0.5):
        super(KnowledgeRNN, self).__init__()

        self.ntokens = ntokens
        self.add_module('drop', nn.Dropout(dropout))

        self.input_embed_size = input_embed_size
        self.add_module("encoder", nn.Embedding(ntokens, input_embed_size))

        query_input_size = state_size + input_embed_size
        self.add_module("query_net", kb.KnowledgeQueryNet(query_input_size,
                                                          hidden_size=query_input_size,
                                                          query_size=query_size,
                                                          value_size=value_size,
                                                          kb=knowledge_base))
        self.lstm_input_size = input_embed_size + self.query_net.value_size

        self.state_size = state_size
        self.add_module("lstm_layer", nn.LSTMCell(self.lstm_input_size, self.state_size))

        self.decoder_input_size = self.state_size + input_embed_size + self.query_net.value_size
        self.add_module("decoder", nn.Linear(self.decoder_input_size, ntokens))

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
        self.query_net.init_weights()

    def forward(self, input):
        batch_size=1
        # input = torch.randn(seqlength, batch_size, input_embed_size)

        lstm_states = []
        kb_vals = []
        encoded = self.drop(self.encoder(input))
        emb = encoded.view(-1,1, self.input_embed_size)

        hx, cx = self.init_hidden()
        for i in emb:
            squeezed_hx = hx.view(self.state_size)
            squeezed_i = i.view(self.input_embed_size)
            query_val = self.query_net(torch.cat((squeezed_hx,
                                                  squeezed_i)).to(device))

            lstm_input = torch.cat((squeezed_i, query_val)).view(1,-1)
            hx, cx = self.lstm_layer(lstm_input, (hx, cx))

            lstm_states.append(squeezed_hx)
            kb_vals.append(query_val)

        lstm_output = self.drop(torch.cat(lstm_states).to(device)).view(-1, self.state_size)
        kb_output = self.drop(torch.cat(kb_vals).to(device)).view(-1, self.query_net.value_size)
        output = torch.cat((encoded, kb_output, lstm_output), dim=1)
        decoded = self.decoder(output)

        # decoded = decoded.view(-1, self.ntokens)
        return F.log_softmax(decoded, dim=1)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.query_net = self.query_net.to(*args, **kwargs)
        return self

    def init_hidden(self):
        # todo: investigate this thing
        weight = next(self.parameters())

        return (weight.new_zeros(1, self.state_size),
                weight.new_zeros(1, self.state_size))

        #return (weight.new_zeros(1, 1, self.state_size),
        #        weight.new_zeros(1, 1, self.state_size))


# we somehow need to tokenize and then
criterion = nn.NLLLoss()
#criterion = nn.CrossEntropyLoss()

def train(model, train_data, lr, doc_size=1000):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()

    for iteration, doc in enumerate(itertools.islice(train_data, 300)):
        doc = doc[:doc_size]
        gc.collect()
        # train_data documents must contain end symbol
        # don't predict on end symbol and don't predict the first symbol
        input = doc[:-1]
        targets = doc[1:]
        # we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()

        output = model(input)

        #print(output.shape, targets.shape)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        # to do: do we iterate over them all here??

        for p in model.parameters():
            if p.requires_grad:
                p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item() / len(doc)

        LOG_INTERVAL = 100
        if iteration % LOG_INTERVAL == 0:
            cur_loss = total_loss / LOG_INTERVAL
            elapsed = time.time() - start_time
            print('| {:5d} | {:5d} tokens | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                iteration, len(doc) , lr,
                elapsed * 1000 / LOG_INTERVAL, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(data_source, model, idx2word):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for iteration, doc in enumerate(itertools.islice(data_source, 1000)):
            doc = doc[:500]
            gc.collect()
            input = doc[:-1]
            targets = doc[1:]
            output = model(input)
            total_loss += criterion(output, targets).item() / len(input)

        print(' '.join([idx2word[t] for t in targets[:45]]))
        print(' '.join([idx2word[i] for i in torch.argmax(output, dim=1)[:45]]))

    return total_loss


def run(epochs=100,
        state_size=50,
        input_embed_size=100,
        value_size=80,
        query_size=40,
        lr=20,
        data_path='./data/wikitext-2-raw/wiki.{0}.raw'):
    best_val_loss = None
    corpus=data.Corpus(data_path)

    ntokens = len(corpus.dictionary)
    train_data = corpus.get_docs('train', device)
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        model = KnowledgeRNN(ntokens,
                             state_size,
                             input_embed_size,
                             value_size=value_size,
                             query_size=query_size).to(device)

        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            train(model, train_data, lr)
            print('Running evaluation...')
            val_loss = evaluate(corpus.get_docs('valid', device), model, corpus.dictionary.idx2word)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open('model.pt', 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

data_path_103 = './data/wikitext-103-raw/wiki.{0}.raw'

if __name__ == '__main__':
    run(epochs=2,
        state_size=10,
        input_embed_size=10,
        query_size=10,
        value_size=10)
