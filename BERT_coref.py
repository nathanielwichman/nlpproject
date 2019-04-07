import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, SequenceLabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import PretrainedBertIndexer

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

from prepare_ROC import prepareBERT

torch.manual_seed(1)

A_VAL = 1
B_VAL = 2
PRO_VAL = 3
OTHER = 4

class ROCReader(DatasetReader):
    def __init__(self):
        super().__init__(lazy=False)
        self.token_indexers = {"tokens": PretrainedBertIndexer(
            pretrained_model = "bert-base-cased",
            do_lowercase = False
            #max_pieces=config.max_seq_length
        )}

    def text_to_instance(self, sentence, A, B, pronoun, answer = None):
        tokens = [Token(word) for word in sentence]
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}
        labels = list()
        # set up labels
        index = 0  # to deal with tokens being split
        # note, depending on initial indexing may need to change for
        # multi word tokens
        for i in range(len(tokens)):
            if sentence[i][0] == '#':
                index -= 1
            if (index >= A[1] and index <= A[2]):
                labels.append(A_VAL) #A
            elif (index >= B[1] and index <= B[2]):
                labels.append(B_VAL) #B
            elif (index >= pronoun[1] and index <= pronoun[2]):
                labels.append(PRO_VAL) #pronoun
            else:
                labels.append(OTHER) # Nothing
            index += 1
        """
        try:
            labels.index(1)
            labels.index(2)
            labels.index(3)
        except ValueError:
            print (sentence)
            print(labels)
            print(A)
            print(B)
            print(pronoun)
            print(answer)
        """

        fields["labels"] = SequenceLabelField(labels, sentence_field)
        if answer is not None:
            fields["answer"] = LabelField(answer, skip_indexing=True)
        return Instance(fields)

    def _read(self, data):
        for d in data:
            yield self.text_to_instance([word for word in self.token_indexers["tokens"].wordpiece_tokenizer(d[0])], d[1], d[2], d[3], d[4])

# dimentions:
# get output >> choose rep tokens >> cross product >> classify
# bert token size >> bert token size >> 2 * bert^2 + bert >> 2

class BERTWino(Model):
    def __init__(self, word_embeddings, vocab):
        super().__init__(vocab)
        self.hiddendim = 50
        self.word_embeddings = word_embeddings
        dims = word_embeddings.get_output_dim()  #768
        # [cross product A : 0 padding : cross product B]
        self.chooser = torch.nn.Sequential(
            torch.nn.Linear(dims * 3 * 2, self.hiddendim),
            torch.nn.Linear(self.hiddendim, self.hiddendim),
            torch.nn.Linear(self.hiddendim, 2))

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    # batch size, seqence length, dimension
    # ipdb (debugger)
    def forward(self, sentence, labels, answer=None):

        # what to do with mask?



        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)[:, 1:-1, :]

        # compute cross product of objects with the pronoun in question
        # gather function! maybe not
        #print(sentence)

        labels2 = torch.cat([labels[:,:,None]]*embeddings.size()[2], 2)
        mask_tensor = torch.zeros(embeddings.size())
        #print(torch.cat([torch.tensor([0])]*embeddings.size()[2]))

        #test = torch.ones(labels2.shape) * 3


        Amask = torch.where(labels2 == (torch.ones(labels2.shape).long() * A_VAL), embeddings, mask_tensor)
        Aavg = Amask.sum(1) / (labels==A_VAL).sum(1).view(labels.shape[0], 1).float()

        Bmask = torch.where(labels2 == (torch.ones(labels2.shape).long() * B_VAL), embeddings, mask_tensor)
        Bavg = Bmask.sum(1) / (labels == B_VAL).sum(1).view(labels.shape[0], 1).float()

        Promask = torch.where(labels2 == (torch.ones(labels2.shape).long() * PRO_VAL), embeddings, mask_tensor)
        Proavg = Promask.sum(1) / (labels == PRO_VAL).sum(1).view(labels.shape[0], 1).float()

        Ainput = torch.cat([Aavg, Proavg, Aavg * Proavg], 1)
        Binput = torch.cat([Bavg, Proavg, Bavg * Proavg], 1)
        #print(labels)
        #torch.set_printoptions(profile="default")


        #Amatrix = torch.cat(torch.mul(AEmbed, ProEmbed))
        #Bmatrix = torch.mul(BEmbed, ProEmbed)
        #Bmatrix = torch.matmul(embeddings[0][labels.index(1)].unsqueeze(0).transpose(0, 1),
        #                      embeddings[0][labels.index(2)].unsqueeze(0))
        # check

        state = self.chooser(torch.cat([Ainput, Binput], 1))

        output = {"tag_logits": state}
        #output = {}
        #print (answer)
        if answer is not None:
            result = answer.tolist()[0]
            if result == 0:
                answer2 = torch.tensor([1, 0])
            else:
                answer2 = torch.tensor([0, 1])

            self._accuracy(state, answer)

            #print(state)
            #print(answer)
            #print("")
            output["loss"] = self._loss(state, answer)

        return output

    def get_metrics(self, reset):
        return {"accuracy": self._accuracy.get_metric(reset)}

def getexamples(model, sentences):
  print("examples")
  counter = 0
  for i in sentences:
      counter += 1;
      if counter > 6:
          break
      else:
          output = model.forward_on_instance(i)
          print(i)
          print(output)
          print("")



LR = 0.00005
BATCH = 5 #16, 32
EPOCHS = 3 #3, 4

data1, data2  = prepareBERT(350, 150)
print("train size:" + str(len(data1)))
print("test size:" + str(len(data2)))

reader = ROCReader()

train_dataset = reader._read(data1)
#print(train_dataset[0])
validation_dataset = reader._read(data2)

vocab = Vocabulary()#Vocabulary.from_instances(train_dataset + validation_dataset)

bert_embedder = PretrainedBertEmbedder(
        pretrained_model="bert-base-cased",
        top_layer_only=True
        # check Rowan email for other details
)

word_embeddings = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                        allow_unmatched_keys = True)

mymodel = BERTWino(word_embeddings, vocab)

optimizer = optim.Adam(mymodel.parameters(), lr=LR)

iterator = BucketIterator(batch_size=BATCH, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(model=mymodel,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  num_epochs=EPOCHS)
indexer = PretrainedBertIndexer(
            pretrained_model = "bert-base-cased",
            do_lowercase = False
            #max_pieces=config.max_seq_length
        )

trainer.train()

getexamples(mymodel, reader._read(data2))