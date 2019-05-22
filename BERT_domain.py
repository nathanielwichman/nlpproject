from prepare_ROC import get_all_pos

LR = 0.00005
BATCH = 32  # 16, 32
EPOCHS = 1  # 3, 4
INFILE = ""
OUTFILE_MODEL = "BERT_pos_model"
OUTFILE_POS = "BERT_pos_tags.txt"

data, m1, m2 = get_all_pos(1000)
print(m1)
print(m2)
print(len(data))
print(data[0:10])

