import tensorflow as tf
from tqdm import  tqdm
pinyin_list = [];hanzi_list = []
pinyin_vocab = set();hanzi_vocab = set()

max_length = 64

with open("zh.tsv", errors="ignore", encoding="UTF-8") as f:
    context = f.readlines()
    for line in context:
        line = line.strip().split("	")
        pinyin = ["GO"] + line[1].split(" ") + ["END"];hanzi = ["GO"] + line[2].split(" ") + ["END"]
        for _pinyin, _hanzi in zip(pinyin, hanzi):
            pinyin_vocab.add(_pinyin);hanzi_vocab.add(_hanzi)

        pinyin = pinyin + ["PAD"] * (max_length - len(pinyin))
        hanzi = hanzi + ["PAD"] * (max_length - len(hanzi))
        pinyin_list.append(pinyin);hanzi_list.append(hanzi)

pinyin_vocab = ["PAD"] + list(sorted(pinyin_vocab))
hanzi_vocab = ["PAD"] + list(sorted(hanzi_vocab))

#这里截取一部分数据
pinyin_list = pinyin_list
hanzi_list = hanzi_list

def get_dataset():
    pinyin_tokens_ids = []
    hanzi_tokens_ids = []

    for pinyin,hanzi in zip(tqdm(pinyin_list),hanzi_list):
        pinyin_tokens_ids.append([pinyin_vocab.index(char) for char in pinyin])
        hanzi_tokens_ids.append([hanzi_vocab.index(char) for char in hanzi])
    #len(pinyin_vocab): 1154
    #len(hanzi_vocab): 4462
    return pinyin_vocab,hanzi_vocab,pinyin_tokens_ids,hanzi_tokens_ids




if __name__ == '__main__':

    pinyin_vocab,hanzi_vocab,pinyin_tokens_ids,hanzi_tokens_ids = get_dataset()
    print(len(pinyin_vocab))
    print(len(hanzi_vocab))
