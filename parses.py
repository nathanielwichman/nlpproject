import json
import spacy
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.util import get_spacy_model

spacy_ = get_spacy_model('en_core_web_sm', pos_tags=True, parse=True, ner=False)

# pass whcih = 0 for both parses, 1 for just depend, 2 for just const
def parse(data, which=0):
    if which != 2:
        # depend parse
        darchive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
        dpred = Predictor.from_archive(darchive, 'biaffine-dependency-parser')

    if which != 1:
        # const parse
        carchive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
        cpred = Predictor.from_archive(carchive, 'constituency-parser')

    for d in data:
        if which != 2:
            dep = dpred.predict_json({"sentence": d.sentence})
            d.depend = dep
        if which != 1:
            con = cpred.predict_json({"sentence": d.sentence})
            d.const = con

def name_rec(data):
    #archive = load_archive("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.28.tar.gz")
    #predictor = Predictor.from_archive(archive, 'ner-model')
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    for d in data:
        d.ner_parse = predictor.predict(sentence=d.sentence)

# Given a list of structs, applies spacy pos
def POStags(sentences):
    nlp = spacy.load('en_core_web_sm')
    for s in sentences:
        s.pos = nlp(s.sentence)
    


def split_sentence(s, c = None):
    toreturn = spacy_(s)
    return toreturn
    if s in c:
        return c[s]
    #pred = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
    result = pred.predict(sentence=s)
    c[s] = result['document']
    return result['document']

def corefresolution(data):
    returndata = list()
    pred = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
    #pred = Predictor.from_archive(archive, 'coref-model')
    count = 0
    for d in data:

        try:  # if all words < 5 letters, exception thrown
            result = pred.predict(document=d.sentence)
            d.coref = result
            returndata.append(d)
            #print (d.sentence)
            #print (result['document'])
        except:
            count += 1
    print("had " + str(count) + " exceptions raised")
    return returndata
