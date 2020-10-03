import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import itertools
import data

torch.manual_seed(50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# todo:
# device, check backpropagating, batches, valid vs test


class KnowledgeRNN(nn.Module):

    def __init__(self, ntokens, state_size, input_embed_size, dropout=0.5):
        super(KnowledgeRNN, self).__init__()
        self.ntokens = ntokens
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntokens, input_embed_size)
        self.total_hidden_size = state_size
        self.lstm_layer = nn.LSTMCell(input_embed_size, self.total_hidden_size)
        self.decoder = nn.Linear(self.total_hidden_size, ntokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input):
        batch_size=1
        # input = torch.randn(seqlength, batch_size, input_embed_size)

        outputs = []
        encoded = self.encoder(input)
        emb = torch.unsqueeze(self.drop(encoded), 1)

        hx, cx = self.init_hidden()
        for i in emb:
            hx, cx = self.lstm_layer(i, (hx, cx))
            outputs.append(hx)

        output = self.drop(torch.cat(outputs).to(device))
        decoded = self.decoder(output)

        decoded = decoded.view(-1, self.ntokens)
        return F.log_softmax(decoded, dim=1)



    def init_hidden(self):
        # todo: investigate this thing
        weight = next(self.parameters())

        return (weight.new_zeros(1, self.total_hidden_size),
                weight.new_zeros(1, self.total_hidden_size))

        #return (weight.new_zeros(1, 1, self.total_hidden_size),
        #        weight.new_zeros(1, 1, self.total_hidden_size))


# we somehow need to tokenize and then
criterion = nn.NLLLoss()
#criterion = nn.CrossEntropyLoss()

def train(model):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()

    for iteration, doc in enumerate(itertools.islice(train_data, 1000)):
        gc.collect()
        # train_data documents must contain end symbol
        # don't predict on end symbol and don't predict the first symbol
        input = doc[:-1]
        targets = doc[1:]
        # we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()

        output = model(input)

        print(output.shape, targets.shape)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        for p in model.parameters():
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


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for iteration, doc in enumerate(itertools.islice(data_source, 1000)):
            gc.collect()
            input = doc[:-1]
            targets = doc[1:]
            output = model(input)
            total_loss += criterion(output, targets).item() / len(input)
    return total_loss


lr = 20
best_val_loss = None
epochs = 10
#corpus = data.Corpus('./data/wikitext-103-raw/wiki.{0}.raw')
corpus = data.Corpus('./data/wikitext-2-raw/wiki.{0}.raw')
corpus = Corpus('./wikitext-2-raw/wiki.{0}.raw')
ntokens = len(corpus.dictionary)
train_data = corpus.get_docs('train', device)
# At any point you can hit Ctrl + C to break out of training early.
try:
    state_size = 100
    input_embed_size = 50
    model = KnowledgeRNN(ntokens, state_size, input_embed_size).to(device)

    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train(model)
        print('Running evaluation...')
        val_loss = evaluate(corpus.get_docs('valid', device))
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
