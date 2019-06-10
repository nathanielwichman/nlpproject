INPUT_FILES = ["BERT_pos_phase1.txt"]
OUTPUT_FILE = "BERT_pos_phase2.txt"

def process_line(line, o):
    end = len(line)
    i = 0
    while i < end:
        o.write(line[i])
        if line[i] == ".":
            o.write("\n")
            i += 1 # skip following space
        i += 1
    o.write("\n")

test_limit = 30
o = open(OUTPUT_FILE, 'w')
for f in INPUT_FILES:
    with open(f) as file:
        for line in file:
            process_line(line, o)
o.close()


