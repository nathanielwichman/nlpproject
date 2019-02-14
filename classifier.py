import torch
import csv
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from allennlp.modules.elmo import Elmo, batch_to_ids
from prepare_wino import process
torch.manual_seed(1)

class WinoClassifier(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(WinoClassifier, self).__init__()

        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        self.embedding = Elmo(options_file, weight_file, 2, dropout=0)
        embedding_dim = 1024
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers)
        self.dropout = nn.Dropout(0.4)
        self.out = nn.Linear(hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input, hidden):
        embedded = self.embedding(input)['elmo_representations']
        embedded_tensor = torch.tensor(embedded[0][0]).unsqueeze(0)
        result, hidden = self.gru(embedded_tensor, hidden)
        result = self.dropout(result)
        return self.sigmoid(self.out(result)), hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

def test(classifier, data, loss_function):
    with torch.no_grad():
        loss = 0
        for sentence, classval in data:
            hidden = classifier.init_hidden()
            for i in range(len(sentence)):
                output, hidden = classifier(batch_to_ids(sentence[i]), hidden)
            answer_tensor = torch.FloatTensor([[classval]])
            loss += loss_function(output, answer_tensor)
        return loss


def train(classifier, sentence, answer, optimizer, loss_function):
    classifier.zero_grad()

    hidden = classifier.init_hidden()
    for i in range(len(sentence)):
        output, hidden = classifier(batch_to_ids(sentence[i]), hidden)

    answer_tensor = torch.FloatTensor([[[answer]]])

    loss = loss_function(output, answer_tensor)
    loss.backward()
    optimizer.step()
    return loss


HIDDEN_SIZE = 50
NUM_LAYERS = 1
ALPHA = .0001
EPOCHS = 10
SIZE = 1000
TEST_PER = 20
PATH = "classifier_test_1.pt"
READ = True

if READ:
    model = WinoClassifier(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    data = list()

    num_examples
    with open('ROCwi17.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        total_count = 0
        for row in csv_reader:
            if line_count > 1:
                for i in range(2,6):
                    if total_count > num_examples:
                        break
                    data.append((row[i].split(), 0))
                    total_count += 1
            line_count += 1


test_index = int(SIZE / (100 / TEST_PER))
data = process(SIZE)
test_data = data[:test_index]
data = data[int(SIZE/5):]
print ("train size: " + str(len(data)))
print ("test size: " + str(len(test_data)) + " / " + str(test_index))
classifier = WinoClassifier(HIDDEN_SIZE, NUM_LAYERS)
optimizer = optim.Adam(classifier.parameters(), ALPHA)
loss_function = nn.BCELoss()

print("starting training...")
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i in range(len(data)):
        if (i % 50 == 0):
            print (i)
        loss = train(classifier, data[i][0], data[i][1], optimizer, loss_function)
        epoch_loss += loss
    test_loss = test(classifier, test_data, loss_function)
    print("EPOCH " + str(epoch) + ": train = " + str(epoch_loss / len(data)) + ", test = " + str(test_loss / len(test_data)))
    sys.stdout.flush()

torch.save(classifier.state_dict(), PATH)
