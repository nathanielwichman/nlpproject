import json
import xml.etree.ElementTree as ET
from statistics import mean, stdev
from parses import parse, corefresolution
from process_wino_xml import getwino, getswitch

# basic data structure for storing information about a sentence
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
        self.nameindexes = None # for ROCname recog
        self.switch = None  # switch word, for future use

# prints some stats to a file
def examine(filename, data):
    #f = open(filename, 'w')

    u_range = list()
    a_range = list()
    b_range = list()
    length = list()
    count = 0
    for d in data:

        formatted = list()
        if "%$" in d.sentence:
            print("UNIQUE ID STRING ALREADY IN INPUT!!!\n\n\n")
        # tag important words, necessary incase of repeated words

        for i in range(len(d.sentence)):

            if i == d.a[1]:
                formatted.append("%")
                formatted.append("$")
                formatted.append("a")
            if i == d.b[1]:
                formatted.append("%")
                formatted.append("$")
                formatted.append("b")
                #formatted.append(" ")
            if i == d.pronoun[1]:
                formatted.append("%")
                formatted.append("$")
                formatted.append("u")
                #formatted.append(" ")
            formatted.append(d.sentence[i])

        # get index of tagged words
        #print (formatted)
        #print (d.pronoun[1])
        f_split = "".join(formatted).split()
        a_i = [i for i, word in enumerate(f_split) if word.startswith('%$a')]
        b_i = [i for i, word in enumerate(f_split) if word.startswith('%$b')]
        u_i = [i for i, word in enumerate(f_split) if word.startswith('%$u')]
    
        count += 1
        if len(a_i) == 0 or len(b_i) == 0 or len(u_i) == 0:
            print ("issue: " + str(count))
            print("".join(formatted))
            continue
        continue
        head = d.depend['predicted_heads'].index(0)

        a_range.append(a_i[0] - head)
        b_range.append(b_i[0] - head)
        u_range.append(u_i[0] - head)
        length.append(len(d.sentence.split()))
    all_range = a_range + b_range
    #print(a_range)
    print("name a: " + str(mean(a_range)) + " (dev: " + str(stdev(a_range)) + ")")
    print("name b: " + str(mean(b_range)) + " (dev: " + str(stdev(b_range)) + ")")
    print("a + b : " + str(mean(all_range)) + " (dev: " + str(stdev(all_range)) + ")")
    print("? pos : " + str(mean(u_range)) + " (dev: " + str(stdev(u_range)) + ")")
    print("length: " + str(mean(length)) + " (dev: " + str(stdev(length)) + ")")


# prints out some info to a file
def output(filename, data):
    f = open(filename, "w")
    for d in data:
        f.write(d.sentence + "\r\n")
        f.write(str(d.coref))
        continue

        # write annotated sentence
        formatSentence = list()
        sentence = d.sentence
        print("writing: " + sentence)
        for i in range(len(sentence) + 1):
            if i == d.a[1] or i == d.b[1]:  # tag choices
                formatSentence.append("(")
            elif i == (d.a[2] + 1) or i == (d.b[2] + 1):
                formatSentence.append(")")
            elif i == d.pronoun[1]:
                formatSentence.append("[")
            elif i == d.pronoun[2] + 1:
                formatSentence.append("]")
            if i < len(sentence):
                formatSentence.append(sentence[i])
        f.write("".join(formatSentence) + "\r\n")

        # write depend
        f.write("dependency parse:\r\n")
        f.write(str(d.depend['pos']) + "\r\n")
        f.write(str(d.depend['predicted_heads']) + "\r\n")

        # write coref
        f.write("constituency parse parse:\r\n")
        f.write(str(d.const['pos_tags']) + "\r\n")

        f.write("\r\n\r\n")
    f.close()

def outputcoref(sentenceData):
    labels = ["didn't find prnoun", "correct answer", "incorrect answer", "thought both were the same", "neither a nor b selected"]
    cases = [list(), list(), list(), list(), list()]

    for d in sentenceData:
        coref = d.coref['clusters']
        if len(coref) < 1:
            print ("< no clusters found, ignoring")
            print (d.sentence)
            continue


        for i in range(len(coref)):
            # get names in cluster to prevent index issues
            clusternames = [" ".join(d.coref['document'][a[0]:a[1]+1]) for a in coref[i]]


            if [d.pronoun[1], d.pronoun[2]] not in coref[i]:
                if i == len(coref) - 1:
                   #print (d.sentence)
                   #print (d.pronoun)
                   #print (coref)
                   cases[0].append(d)
                else:
                    continue

            #elif [d.a[1], d.a[2]] in coref[i] and [d.b[1], d.b[2]] in coref[i]:
            elif d.a[0] in clusternames and d.b[0] in clusternames:
                cases[3].append(d)
            #elif ([d.a[1], d.a[2]] in coref[i] and d.answer =='A') or \
            #        ([d.b[1], d.b[2]] in coref[i] and d.answer == 'B'):
            elif d.a[0] in clusternames and d.answer == 'A' or \
                    d.b[0] in clusternames and d.answer == 'B':
                cases[1].append(d)
            #elif ([d.a[1], d.a[2]] in coref[i] and d.answer =='B') or \
            #       ([d.b[1], d.b[2]] in coref[i] and d.answer == 'A'):
            elif d.a[0] in clusternames and d.answer == 'B' or \
                    d.b[0] in clusternames and d.answer == 'A':
                cases[2].append(d)
            else:
               cases[4].append(d)
            break
    return labels, cases

# given a filenum, returns the raw text (since the format is basically empty)
def getSentence(filenum):
    f = open('WinoCoref/WinoCoref/data/' + filenum + '.sgm')
    next = False
    for line in f:
        strippedline = line.rstrip("\n\r")
        if next:
            return strippedline
        if strippedline == "<TEXT>":
            next = True


trainFiles = list()
fp = open('WinoCoref/WinoCoref/train.txt', 'r')
for line in fp:
    trainFiles.append(line.rstrip("\n\r"))

# structure data for each sentence, and add to list
sentenceData = list()
count = 1  # track index to check for formatting trends
for x in trainFiles:
    # ignore extraneous files
    if len(x) <= 8 or x[-8:] != ".apf.xml":
        continue

    # apply cutoff for limited dataset
    cutoff = 100
    if (count > cutoff):
        break

    data = struct()
    # Get xml ready to read
    tree = ET.parse('WinoCoref/WinoCoref/data/' + x)
    root = tree.getroot()
    info = struct()

    # get raw text
    data.sentence = getSentence(x[:-8])

    # read in info for each special word
    sentences = root.findall("document/entity/entity_mention")

    # don't know why this is needed, but it is
    offset = 10
    if count >= 11:
        offset = 11
    if count >= 102:
        offset = 12

    # Add raw text + start and end indexes to struct
    pronoun = sentences[0].find("head/charseq")
    data.pronoun = (pronoun.text, int(pronoun.get("START")) - offset, int(pronoun.get("END")) - offset)

    a = sentences[1].find("head/charseq")
    data.a = (a.text, int(a.get("START")) - offset, int(a.get("END")) - offset)

    b = sentences[2].find("head/charseq")
    data.b = (b.text, int(b.get("START")) - offset, int(b.get("END")) - offset)
    #print(str(data.pronoun) + ": " + str(data.a) + ", " + str(data.b))
    # add to list
    sentenceData.append(data)
    count += 1

def replace(d, indices):
    result = []

    for i in indices:
        #print(i)
        #print(str(i[0]) + ", " + str(i[1]))
        replace = " ".join(d.coref["document"][i[0]:i[1]+1]) + "(" + str(i[0]) + ", " + str(i[1]) + ")"
        result.append(replace)
    #print(" asd " + replace)
    return result

def Switch_Words():
    results = getswitch()
    for item in results:
        print(str(item[0]) + ": " + str(item[1]))
    exit()

def BERT_Wino():
    sentenceData = corefresolution(getwino())
    return sentenceData

# Checks list of stucts with allennlp coref and outputs answer and stats
def Check_Wino(inputdata):
    #Switch_Words()
    #parse(sentenceData)
    sentenceData = corefresolution(inputdata)

    #examine("test_stats22", sentenceData)
    #output("test_coref", sentenceData)
    labels, cases = outputcoref(sentenceData)


    """
    for correct, incorrect in zip(cases[1][:12], cases[2][:12]):
        print("correct")
        print(correct.sentence)
        #print(correct.coref)
        print(correct.coref['top_spans'])
        print(correct.coref['clusters'])
        print(correct.coref['chances'])
        print("\nincorrect")
        print(incorrect.sentence)
        #print(incorrect.coref)
        print(incorrect.coref['top_spans'])
        print(incorrect.coref['clusters'])
        print(incorrect.coref['chances'])
        print("\n\n")
    """


    print("")
    # for averaging confidence of correct/incorrect answers
    correct = []
    correct_other = []
    wrong = []
    wrong_other = []
    A_count = 0
    B_count = 0

    for i in range(len(labels)):
        label = labels[i]
        case = cases[i]
        print(label)
        for j in range(min(1200, len(case))):
            # calculate avg confidence

            if i == 1 or i == 2:
                pronoun_index = case[j].coref['top_spans'].index([case[j].pronoun[1], case[j].pronoun[2]])


                scores = case[j].coref['chances2'][pronoun_index]
                #print (scores)
                if case[j].nameindexes == None:
                    try:
                        A_index = case[j].coref['top_spans'].index([case[j].a[1], case[j].a[2]])

                        A_index = pronoun_index - A_index
                        #print(A_index)
                        A_score = scores[A_index]

                    except:
                        A_score = float("-inf")

                    try:
                        B_index = case[j].coref['top_spans'].index([case[j].b[1], case[j].b[2]])
                        B_index = pronoun_index - B_index
                        B_score = scores[B_index]

                    except:
                        B_score = float("-inf")
                else:  # consider multiple name cases
                    A_indexes = case[j].nameindexes[1][case[j].a[0]]  # get all indices of A
                    A_score = float("-inf")
                    for ind in A_indexes:
                        if ind == case[j].pronoun[1]:
                            continue
                        try:

                            A_index = case[j].coref['top_spans'].index([ind, ind])


                            A_index = pronoun_index - A_index
                            #print(case[j].coref["document"][ind])
                            A_score = max(A_score, scores[A_index]) # find best A score


                        except:
                            A_score = max(A_score, float("-inf"))

                    B_indexes = case[j].nameindexes[1][case[j].b[0]]  # get all indices of A
                    B_score = float("-inf")

                    for ind in B_indexes:
                        if ind == case[j].pronoun[1]:
                            continue
                        try:
                            B_index = case[j].coref['top_spans'].index([ind, ind])

                            B_index = pronoun_index - B_index
                            # print(A_index)
                            B_score = max(B_score, scores[B_index])  # find best A score

                        except:
                            B_score = max(B_score, float("-inf"))


                case[j].A_score = A_score
                case[j].B_score = B_score
                if i == 1: # correct
                    if case[j].answer == 'A':
                        if B_score != float("-inf"):
                            correct_other.append(B_score)
                        case[j].correct_correct = A_score
                        case[j].correct_wrong = B_score
                    else:
                        if A_score != float("-inf"):
                            correct_other.append(A_score)
                        case[j].correct_correct = B_score
                        case[j].correct_wrong = A_score
                else: # incorrect
                    if case[j].answer == 'A':
                        if A_score != float("-inf"):
                            wrong_other.append(A_score)
                        case[j].wrong_correct = A_score
                        case[j].wrong_wrong = B_score
                    else:
                        if B_score != float("-inf"):
                            wrong_other.append(B_score)
                        case[j].wrong_correct = B_score
                        case[j].wrong_wrong = A_score

                for s in case[j].coref['clusters']:
                    if [case[j].pronoun[1], case[j].pronoun[2]] in s:
                        coref = case[j].coref
                        if i == 1: # correct
                            correct.append(coref['chances'][coref['top_spans'].index(s[1])])
                        else:
                            wrong.append(coref['chances'][coref['top_spans'].index(s[1])])


            print(case[j].sentence)
            print("pronoun: " + str(replace(case[j], [[case[j].pronoun[1], case[j].pronoun[2]]])), end='')

            print(", A: " + str(replace(case[j], [[case[j].a[1], case[j].a[2]]])), end='')
            print(", B: " + str(replace(case[j], [[case[j].b[1], case[j].b[2]]])))
            #print(case[j].coref['top_spans'])
            print("clusters: [", end='')
            for c in case[j].coref['clusters']:
                print(replace(case[j], c), end='')
            print("]")
            #+ str(replace(case[j], case[j].coref['clusters'])))

            if i == 1 or i == 2:
                if (case[j].A_score > case[j].B_score):
                    A_count += 1
                else:
                    B_count += 1
                answer = ""
                if case[j].answer == "A":
                    answer = case[j].a[0]

                else:

                    answer = case[j].b[0]
                print("Winograd answer: " + str(case[j].answer) + " (" + answer + ")")
                print("A confidence: " + str(case[j].A_score))
                print("B confidence: " + str(case[j].B_score))

            #if i == 1: # correct

            #   print ("confidence for choice: " + str(case[j].correct_correct))
            #   print ("confidence for other : " + str(case[j].correct_wrong))
            #if i == 2:
            #   print ("confidence for choice: " + str(case[j].wrong_wrong))
            #    print ("confidence for other : " + str(case[j].wrong_correct))



            #print(case[j].coref['predicted_antecedents'])
            #print(case[j].coref['pre'])
            print("")
        print("\n\n\n")

    for i in range(len(cases)):
        print(labels[i])
        print(len(cases[i]))

    print("")

    overall = correct + wrong_other + wrong + correct_other
    overall_correct = correct + wrong_other
    overall_wrong = wrong + correct_other
    print("overall confidence: " + str(mean(overall)) + ", " + str(
        stdev(overall)))
    print("overall confidence for correct answer   : " + str(mean(overall)) + ", " + str(stdev(overall_correct)))
    print("overall confiidence for incorrrect answer:" + str(mean(overall_wrong)) + ", " + str(stdev(overall_wrong)))
    print("correct confidence: " + str(mean(correct)) + ", " + str(stdev(correct)))
    print("correct other confidence: " + str(mean(correct_other)) + ", " + str(stdev(correct_other)))
    print("incorrect confiden: " + str(mean(wrong)) + ", " + str(stdev(wrong)))
    print("incorrect other confidence: " + str(mean(wrong_other)) + ", " + str(stdev(wrong_other)))
    print("A count: " + str(A_count) + ", B count: " + str(B_count))

# By default, run on the wino set
#Check_Wino(getwino())