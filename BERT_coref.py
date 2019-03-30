import torch
import torch.optim as optim
import numpy as np

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.fields import SpanField

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


torch.manual_seed(1)

class ROCReader(DatasetReader):
    def __init__(self):
        super().__init__(lazy=False)
        self.token_indexers = {"tokens": PretrainedBertIndexer(
            pretrained_model = "bert-base-cased",
            do_lowercase = False
            #max_pieces=config.max_seq_length
        )}

    def text_to_instance(self, sentence, A, B, pronoun, answer = None):
        tokens = [Token(word) for word in self.token_indexers["tokens"].wordpiece_tokenizer(sentence)]
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}
        labels = list()
        # set up labels for each inded
        for i in range(len(tokens)):

            if (i >= A[1] and i <= A[2]):
                labels.append(0) #A
            elif (i >= B[1] and i <= B[2]):
                labels.append(1) #B
            elif (i >= pronoun[1] and i <= pronoun[2]):
                labels.append(2) #pronoun
            else:
                labels.append(3) # Nothing

        fields["labels"] = SequenceLabelField(labels, sentence_field)
        if answer:
            fields["answer"] = LabelField(answer, skip_indexing=True)
        return Instance(fields)

    def _read(self, data):
        for d in data:
            yield self.text_to_instance(d[0], d[1], d[2], d[3])

# dimentions:
# get output >> choose rep tokens >> cross product >> classify
# bert token size >> bert token size >> 2 * bert^2 + bert >> 2

class BERTWino(Model):
    def __init__(self, word_embeddings, vocab):
        super().__init__(vocab)
        self.hiddendim = 200
        self.word_embeddings = word_embeddings
        dims = word_embeddings.get_output_dim()  #768
        # [cross product A : 0 padding : cross product B]
        self.chooser = torch.nn.Sequential(
            torch.nn.Linear(dims * dims * 2, self.hiddendim),
            torch.nn.Linear(self.hiddendim, self.hiddendim),
            torch.nn.Linear(self.hiddendim, 2))

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, sentence, labels, answer=None):
        embeddings = self.word_embeddings(sentence)


        print(self.word_embeddings.get_output_dim())
        print(sentence)
        #print(sentence["tokens-offsets"].size())
        #print(labels.size())

        exit()
        # grab word indexes



LR = 0.00005
BATCH = 32
EPOCHS = 3

data1 = [["Bob 's cat went to the store with Joe . He had to be bribed to come after all",
        ("Bob", 0, 0), ("Joe", 7, 7), ("He", 8, 8)]]
data2 = [["Sally went to the store with Mei . She had to be bribed to come",
        ("Sally", 0, 0), ("Mei", 6, 6), ("She", 8, 8)]]
reader = ROCReader()

train_dataset = reader.read(data1)
validation_dataset = reader.read(data2)

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

trainer.train()

indexer = PretrainedBertIndexer(
            pretrained_model = "bert-base-cased",
            do_lowercase = False
            #max_pieces=config.max_seq_length
        )
print(indexer.wordpiece_tokenizer(data1[0][0]))
