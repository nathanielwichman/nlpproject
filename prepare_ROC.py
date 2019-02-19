
import csv
import random

from parses import name_rec, POStags
from process import Check_Wino

ROC_FILE = "ROCwi17.csv"  # file with rocstories
PARSE_FILE_WRITE = "junk.txt"
PARSE_FILE_READ = "ROC_parses2.txt"

# basic data structure for storing information about a sentence
# for winograd parsing
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
       self.nameindexes = None  # for sentences where a name appear multiple times
       self.switch = None  # switch word, for future use

# Holds information useful for nertagging
class nerstruct:
   sentence = None  # Sentence, as string
   parse = None  # List of words from parsed sentence, per standard format
   tags = None  # Ner parse, indexes coorispond to words
   ner_parse = None  # Full NER parse, not guaranteed to be non-None
   names = None  # Names found in ner parse (list of names, dict from names to indexes)
   pronouns = None # Pronouns found (list of pronouns, dict from names to indexes)
   replaced = False  # if ambiguity had been added, will be true and the below fields will be filled
   pronounindex = None  # switched pronoun
   answer = None  # ground truth
   pos = None  # Part Of Speech tags, for helping to turn name->correct pronoun

   # format in a printable form for debugging
   def string(self):
       return self.sentence + "\n" + str(self.parse) + "\n" + str(self.tags) + "\n"

   # returns the first index where both names have been introduced
   # requires names to be initialized first
   def firstinstance(self):
       if self.names == None:
           return -1
       maxindex = 0
       for n in self.names[0]:
           maxindex = max(maxindex, self.names[1][n][0])
       return maxindex

"""
I/O Stuff
"""
# Prints parsed nerstructs to disk, to be read later.
# Overwrites all prior data in filename
def save(filename, data):
   writer = open(filename, "w")
   for d in data:
       writer.write(d.sentence + "\n")
       writer.write(str(d.parse) + "\n")
       writer.write(str(d.tags) + "\n")

# Restores saved parsed nerstructs from filename, and returns a list of number
# of them
def load(filename, number):
   returndata = list()
   with open(filename) as f:
       counter = 0  # for tracking lines
       storedstruct = nerstruct()
       for line in f:
           if counter % 3 == 0:  # first line, raw text
               storedstruct.sentence = line.rstrip()
           elif counter % 3 == 1:  # second line, parsed words
               storedstruct.parse = readarray(line.rstrip())
           else:  # third line, NER tags
               storedstruct.tags = readarray(line.rstrip())
               returndata.append(storedstruct)

               if counter / 3 >= number:  # read number structs
                   break

               storedstruct = nerstruct()  # set up new clean struct for next loop
           counter += 1
   return returndata

# Given a printed array of strings, returns the array
def readarray(line):
   returnarray = list()
   templist = line[1:len(line) - 1].split(", ")  # get rid of brackets and split on comma

   # parse templist and append words without quotation marks to returnarray
   for d in templist:
       returnarray.append(d[1:len(d) - 1])
   return returnarray

"""
End I/O Stuff
"""


# Gets number ROC stories from given file and returns a list of
# those stories as strings.
def getROC(filename, number):
   # Grabs sentences from CSV file
   sentence_data = list()

   with open(filename) as csv_file:
       csv_reader = csv.reader(csv_file, delimiter=",")
       line_count = 0
       for row in csv_reader:
           if line_count > number:
               break
           if line_count > 1:  # ignore first line
               story = list()
               # move complete story to list
               for i in range(2,6):
                   story += row[i].split()
               # turn into sentences
               sentence_data.append((" ".join(story)).rstrip())
           line_count += 1

   return sentence_data

# Given a set of sentences, tags them with NER and returns list
# of nerstructs with the tagged info.
# Takes a long time
def nertag(data):
   returndata = list()

   # make structs for parsing
   for d in data:
       structdata = nerstruct()
       structdata.sentence = d
       returndata.append(structdata)

   # give structs to be parsed (info added)
   name_rec(returndata)

   # set up fields for easy use
   for d in returndata:
       # take from result dict
       d.parse = d.ner_parse['words']
       d.tags = d.ner_parse['tags']

   return returndata

# Given a nerstruct, returns a list of all unique pronouns as well
# as a dict from pronouns to indexes in the sentence
def getpronouns(sentence):
   pronouns = ["he", "she", "him", "her", "his", "her"]
   pro = set()
   indexes = dict()
   for i in range(len(sentence.parse)):
       word = sentence.parse[i].lower()
       if word in pronouns:  # found a pronoun, update data
           pro.add(word)

           # add list of indexes, or append dif word already found
           if word not in indexes:
               indexes[word] = list()
           indexes[word].append(i)

   return list(pro), indexes

# Given a nerstruct, returns a list of all unique 'U-PER' as well
# as a dict from names to indexes in the sentence
def getnames(sentence):
   names = set()
   indexes = dict()
   # index word by word, adding names found to a set
   for i in range(len(sentence.parse)):
       if sentence.tags[i] == 'U-PER':  # person name
           names.add(sentence.parse[i])  # add associated word

           # add list of indexes, or append if word alredy found
           if sentence.parse[i] not in indexes:
               indexes[sentence.parse[i]] = list()
           indexes[sentence.parse[i]].append(i)

   return list(names), indexes

# Pruning method. Returns true if the sentence fits
# the criteria to be a wino sentence
def criteria(s):

   # check if criteria are met
   if len(s.names[0]) != 2:  # must have exactly 2 names entities
       return False
   elif len(s.pronouns[0]) < 1:  # must have at least 1 pronoun
       return False
   elif s.firstinstance() < 0:
       return False
   else:
       return True

# Prunes a given list of NER parsed sentences to find ones
# that may fulfill Winograd standards
def prune(data):
   pruneddata = list()
   for sentence in data:
       # get names and pronouns
       names, name_dict = getnames(sentence)
       pronouns, pro_dict = getpronouns(sentence)
       # update struct
       sentence.names = (names, name_dict)
       sentence.pronouns = (pronouns, pro_dict)

       if criteria(sentence):
           pruneddata.append(sentence)
   return pruneddata

# Removes names and replaces with generic same gendered pronouns.
# Returns true if usable, false otherwise (data will still be edited)
def removegender(sentence):
   # problem pronouns:
   # her -> him vs his
   #   e.g. "it was her food" vs "i went looking for her"
   #
   #   e.g "asked Bob" vs "Bob asked"
   # she vs her
   malenames = ["Bob", "Fred", "George", "Larry"]
   femalenames = ["Sarah", "Emily", "Liz", "Oprah"]

   # dict for translating pronouns
   mfdict = {
       "him":"her",
       "his":"her",
       "His":"Her",
       "he":"she",
       "He":"She",
       "boy":"girl",
       "Boy":"Girl"

   }
   fmdict = {v: k for k, v in mfdict.items()}
   # choose gender
   nameslist = None
   prodict = None
   gender = ""
   if random.randint(0,1) == 0: # male names
       prodict = fmdict
       nameslist = malenames
       gender = "M"
   else:
       prodict = mfdict
       nameslist = femalenames
       gender = "F"

   # Choose names
   choisea = random.randint(0, len(malenames) - 1)
   namea = nameslist[choisea]
   choiseb = random.randint(0, len(malenames) - 1)

   # make sure names aren't identical
   while choisea == choiseb:
       choiseb = random.randint(0, len(malenames) - 1)
   nameb = nameslist[choiseb]

   # replace names
   for nameindex in sentence.names[1][sentence.names[0][0]]:
       sentence.parse[nameindex] = namea
   for nameindex in sentence.names[1][sentence.names[0][1]]:
       sentence.parse[nameindex] = nameb

   # replace last name with pronoun, if no name first, remove
   found = False
   # making multple results now
   results = list()

   for name in sentence.names[0]:


       # has at least two names referenced
       if len(sentence.names[1][name]) > 1:
           for nameindex in sentence.names[1][name][1:]:
               # confirm both names occur before chosen name
               other = sentence.names[0][0] # get other name
               if other == name:
                   other = sentence.names[0][1]

               otherindex = sentence.names[1][other][0]
               # if other index is after last name (which will be replaced), continue
               #if otherindex > sentence.names[1][name][-1]:
               #print(sentence.sentence)
               #print(name)
               #print(otherindex)
               #print(nameindex)

               if otherindex > nameindex:
                   #print("skipping")
                   #print("")
                   continue
               #print("")

               # result struct
               rs = nerstruct()
               rs.sentence = sentence.sentence
               rs.parse = list(sentence.parse)
               rs.pronouns = sentence.pronouns # assumes this is never hanged
               rs.names = (list(sentence.names[0]), dict(sentence.names[1]))
               rs.pos = sentence.pos

               temp = sentence
               sentence = rs

               # replace last name with pronoun
               if gender == "M":
                   # generalize if pronoun is possesive
                   rs.parse[nameindex] = "he"
                   if sentence.names[1][name][-1] > 0 and \
                           sentence.parse[nameindex - 1] == ".":
                       sentence.parse[nameindex] = "He"
                   if nameindex < len(sentence.parse) - 1 and sentence.parse[nameindex + 1] == "'s":
                       sentence.parse[nameindex] = "his"
                       del sentence.parse[nameindex + 1]
                       if nameindex > 0 and \
                               sentence.parse[nameindex - 1] == ".":
                           sentence.parse[nameindex] = "His"


               else:
                   sentence.parse[nameindex] = "she"
                   if nameindex > 0 and \
                           sentence.parse[nameindex - 1] == ".":
                       sentence.parse[nameindex] = "She"
                   if nameindex < len(sentence.parse) -1 and \
                           sentence.parse[nameindex + 1] == "'s":
                       sentence.parse[nameindex] = "her"
                       del sentence.parse[nameindex + 1]
                       if nameindex > 0 and \
                               sentence.parse[nameindex - 1] == ".":
                           sentence.parse[nameindex] = "Her"
               sentence.replaced = True
               sentence.pronounindex = nameindex

               if sentence.names[0].index(name) == 0:
                   sentence.answer = "A"
               else:
                   sentence.answer = "B"

               # replace pronoun
               for pronoun in sentence.pronouns[0]:
                   if pronoun in prodict:  # loop through each instance of each gendered pronoun
                       for index in sentence.pronouns[1][pronoun]:
                           sentence.parse[index] = prodict[pronoun]  # replace with new gendered form
               sentence.sentence = " ".join(sentence.parse)

               # update struct with new names
               sentence.names[1][namea] = sentence.names[1][sentence.names[0][0]]
               sentence.names[1][nameb] = sentence.names[1][sentence.names[0][1]]
               sentence.names[0][0] = namea
               sentence.names[0][1] = nameb

               results.append(sentence)
               sentence = temp
   return results




# given a list of pruned nerstructs, finds appropriate examples
# and replaces a name with a pronoun
def addambiguity(data):
   returndata = list()
   for d in data:
       callresult = removegender(d)
       #for c in callresult:
       #    print(c.sentence, end='')
       #print("")
       returndata += callresult
       #print (d.parse)
   return returndata

# Takes in a list of pruned nerstructs and prepares and returns a list
# of all possible Winograd structs for each sentence
def prepare(data):
   returnlist = list()
   for sentence in data:
       # loop over all indexes of each pronoun, seeing if they're usable
       if sentence.replaced: # if replaced, just use replaced pronoun
           toadd = struct()
           toadd.nameindexes = sentence.names
           toadd.sentence = sentence.sentence

           aname = sentence.names[0][0]
           bname = sentence.names[0][1]
           aindex = sentence.names[1][aname][0]
           bindex = sentence.names[1][bname][0]
           toadd.a = (aname, aindex, aindex)
           toadd.b = (bname, bindex, bindex)
           toadd.pronoun = (sentence.parse[sentence.pronounindex], sentence.pronounindex, sentence.pronounindex)
           toadd.answer = sentence.answer
           returnlist.append(toadd)
           continue



       nounindex = sentence.firstinstance()
       for p in sentence.pronouns[0]:
           for i in sentence.pronouns[1][p]:
               if i > nounindex:
                   # Setup struct with info
                   toadd = struct()
                   toadd.sentence = sentence.sentence

                   toadd.nameindexes = sentence.names
                   aname = sentence.names[0][0]
                   bname = sentence.names[0][1]
                   aindex = sentence.names[1][aname][0]
                   bindex = sentence.names[1][bname][0]
                   toadd.a = (aname, aindex, aindex)
                   toadd.b = (bname, bindex, bindex)
                   returnlist.append(toadd)
                   if sentence.replaced:  # if we've replaced a name, just update 1 pronoun
                       toadd.pronoun = (sentence.parse[sentence.pronounindex], sentence.pronounindex, sentence.pronounindex)
                       toadd.answer = sentence.answer
                       break
                   else:  # otherwise, just add random answer and keep going
                       toadd.pronoun = (p, i, i)
                       toadd.answer = "A"  # just pick an answer for now

   return returnlist

# Writes the first n ROC stories to disk
def write(n):
   data = getROC(ROC_FILE, n)
   tagged = nertag(data)
   save(PARSE_FILE_WRITE, tagged)

def test():
    data = load(PARSE_FILE_READ, 500)
    pruneddata = prune(data)
    POStags(pruneddata)

    replaced = addambiguity(pruneddata)

    for i in range(min(len(replaced), 50)):
        print(replaced[i].sentence)
        for token in replaced[i].pos:
            print(token.text + "(" + token.pos_ + "/" + token.tag_ + ") ", end="")

        print("")
        print("")
    exit()

    readylist = prepare(replaced)

    """
    for i in range(20):
       print(readylist[i].sentence)
       print(readylist[i].pronoun)
       print(readylist[i].a)
       print(readylist[i].b)
       print("")
    """

    Check_Wino(readylist)

    """
    data = getROC(ROC_FILE, 50)
    parseddata = nertag(data)
    #save(PARSE_FILE_WRITE, parseddata)
    """

test()
