import csv
from random import shuffle

def process(num_examples):
    data = list()


    # get pre-printed data from file
    wino = open("test_output2", 'r')
    wino_count = 0
    for line in wino:
        if wino_count > num_examples:
            wino_count = num_examples
            break
        data.append((line.split(), 1))
        wino_count += 1
    num_examples = min(wino_count, num_examples)

    with open('ROCwi17.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        total_count = 0
        for row in csv_reader:
            if line_count > 1:
                for i in range(2,6):
                    if total_count > num_examples:
                        shuffle(data)
                        return data
                    data.append((row[i].split(), 0))
                    total_count += 1
            line_count += 1

    shuffle(data)
    return data


