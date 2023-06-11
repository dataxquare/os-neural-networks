import torch
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def _one_hot_encode(self, sequence, dict_size, seq_len, batch_size):
        features = np.zeros(
            (batch_size, seq_len, dict_size), dtype=np.float32)

        for i in range(batch_size):
            for u in range(seq_len):
                features[i, u, sequence[i][u]] = 1

        return features

    def encoding(self, text):
        chars = set(''.join(text))
        int2char = dict(enumerate(chars))

        char2int = {char: ind for ind, char in int2char.items()}

        maxlen = len(max(text, key=len))

        for i in range(len(text)):
            while len(text[i])<maxlen:
                text[i] += ' '

        input_seq = []
        target_seq = []

        for i in range(len(text)):
            input_seq.append(text[i][:-1])
            target_seq.append(text[i][1:])

            print("Input Sequence: {}\nTarget Sequence: {}".format(
                input_seq[i], target_seq[i]))

        for i in range(len(text)):
            input_seq[i] = [char2int[character] for character in input_seq[i]]
            target_seq[i] = [char2int[character] for character in target_seq[i]]

        dict_size = len(char2int)
        seq_len = maxlen - 1
        batch_size = len(text)

        input_seq = self._one_hot_encode(
            input_seq, dict_size, seq_len, batch_size)

        input_seq = torch.from_numpy(input_seq)
        target_seq = torch.Tensor(target_seq)

        return input_seq, target_seq, dict_size, seq_len, batch_size, int2char

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)

        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        return hidden

    def predict(model, character, dict_size, int2char):
        character = np.array([[char2int[c] for c in character]])
        character = self._one_hot_encode(character, dict_size, character.shape[1], 1)
        character = torch.from_numpy(character)
        character.to(device)

        out, hidden = model(character)

        prob = nn.functional.softmax(out[-1], dim=0).data
        char_ind = torch.max(prob, dim=0)[1].item()

        return int2char[char_ind], hidden

    def train(self, optimizer, criterion, n_epochs, input_seq, model):
        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            input_seq.to(device)
            output, hidden = model(input_seq)
            loss = criterion(output, target_seq.view(-1).long())
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
                print("Loss: {:.4f}".format(loss.item()))
            
        return model

    def sample(self, model, out_len, start='hey'):
        model.eval()
        start = start.lower()

        chars = [ch for ch in start]
        size = out_len - len(chars)

        for ii in range(size):
            char, h = predict(model, chars)
            chars.append(char)

        return ''.join(chars)
