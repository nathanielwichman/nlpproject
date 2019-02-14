from parses import split_sentence
import random
import xml.etree.ElementTree as ET


class struct:
    def __init__(self):
        self.sentence = None
        self.pronoun = None
        self.a = None
        self.b = None
        self.depend = None
        self.const = None
        self.coref = None
        self.answer = None
        self.switch = None  # switch word, for future
def getswitch():
    wino = ET.parse('WSCollection.xml')
    root = wino.getroot()
    schemas = root.findall("schema")

    returndata = list()
    for schema in schemas:
        sentence_a = schema.find("text/txt1").text.replace("\n", " ").strip()
        pron = schema.find("text/pron").text.replace("\n", "").strip()
        sentence_finish = schema.find("text/txt2").text.replace("\n", "").strip()
        # print(sentence_a)
        full_sentence = (sentence_a + " " + pron + " " + sentence_finish)

        returndata.append(full_sentence.split())

    print("test: " + str(len(returndata)))
    i = 0
    parseddata = list()
    while (i + 1 < len(returndata)):
        diffA = [item for item in returndata[i] if item not in returndata[i + 1]]
        diffB = [item for item in returndata[i + 1] if item not in returndata[i]]
        if (len(diffA) <= 3):
            parseddata.append((diffA, diffB))
        i += 2
    return parseddata

def getwino():
    c = dict()  # for split weirdness
    wino = ET.parse('WSCollection.xml')
    root = wino.getroot()
    schemas = root.findall("schema")

    returndata = list()
    for schema in schemas:
        s = struct()

        sentence_a = schema.find("text/txt1").text.replace("\n"," ").strip()
        pron = schema.find("text/pron").text.replace("\n", "").strip()
        sentence_finish = schema.find("text/txt2").text.replace("\n", "").strip()
        #print(sentence_a)
        full_sentence = (sentence_a + " " + pron + " " + sentence_finish)

        s.sentence = full_sentence
        s.split_sentence = split_sentence(full_sentence, c)
        # get start and end indecies
        #print(split_sentence(sentence_a, c))
        s.pronoun = (pron, len(split_sentence(sentence_a, c)),
                     len(split_sentence(sentence_a, c)) + len(split_sentence(pron, c)) - 1)

        # need to find char index -> word index
        nouns = schema.findall("answers/answer")
        a = nouns[0].text.strip()
        b = nouns[1].text.strip()
        #print(full_sentence)
        #print(nouns[1].text)
        try:
            a_index = full_sentence.index(a)
            b_index = full_sentence.index(b)
        except:
            continue
            print("can't find word")
            print (full_sentence)
            print (a)
            print (b)
            print ("")
            continue

        new_sentence = list()
        for i in range(len(full_sentence)):
            new_sentence.append(full_sentence[i])
            if i == a_index:  # tag start of pronouns
                new_sentence.append("%")
                new_sentence.append("!")
                new_sentence.append("a")
            if i == b_index:
                new_sentence.append("%")
                new_sentence.append("!")
                new_sentence.append("b")
            #new_sentence.append(full_sentence[i])


        newsentence = split_sentence("".join(new_sentence), c)

        #for word in newsentence:
        #   print(word)
        count = 0
        for i in range(len(newsentence)):
            word = str(newsentence[i])

            if len(word) > 1 and word[1:].startswith("%!a"):
                s.a = (a, i, i + len(split_sentence(a, c)) - 1)
                count += 1
            if len(word) > 1 and word[1:].startswith("%!b"):
                s.b = (b, i, i + len(split_sentence(b, c)) - 1)
                count += 278

        if count != 279:
            print ("issue finding tokens")
            print(newsentence)
        else:
            answer = schema.find("correctAnswer").text
            s.answer = answer.strip()[0]
            returndata.append(s)
    #random.shuffle(returndata)
    return returndata
