import random
def age_vocab(max_age, mon=1, symbol=None):
    age2idx = {}
    idx2age = {}
    if symbol is None:
        symbol = ['PAD', 'UNK']

    for i in range(len(symbol)):
        age2idx[str(symbol[i])] = i
        idx2age[i] = str(symbol[i])

    if mon == 12:
        for i in range(max_age):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)
    elif mon == 1:
        for i in range(max_age * 12):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)
    else:
        age2idx = None
        idx2age = None
    return age2idx, idx2age

def code2index(tokens, token2idx, mask_token=None):
    output_tokens = []
    for i, token in enumerate(tokens):
        if token==mask_token:
            output_tokens.append(token2idx['UNK'])
        else:
            output_tokens.append(token2idx.get(token, token2idx['UNK']))
    return tokens, output_tokens


def random_mask(tokens1, tokens2, tokens3, token2idx1,token2idx2,token2idx3, code_map):
    output_label = []
    output_token1 = []
    output_token2 = []
    output_token3 = []
    for i, token in enumerate(tokens3):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token1.append(token2idx1["MASK"])
                output_token2.append(token2idx2["MASK"])
                output_token3.append(token2idx3["MASK"])
            # 10% randomly change token to random token
            elif prob < 0.9:
                tmp=random.choice(list(code_map.keys()))
                output_token1.append(token2idx1[code_map[tmp][1]])
                output_token2.append(token2idx2[code_map[tmp][0]])
                output_token3.append(token2idx3[tmp])
            # -> rest 10% randomly keep current token
            else:
                output_token1.append(token2idx1.get(tokens1[i], token2idx1['UNK']))
                output_token2.append(token2idx2.get(tokens2[i], token2idx2['UNK']))
                output_token3.append(token2idx3.get(tokens3[i], token2idx3['UNK']))
            # append current token to output (we will predict these later
            output_label.append(token2idx3.get(tokens3[i], token2idx3['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token1.append(token2idx1.get(tokens1[i], token2idx1['UNK']))
            output_token2.append(token2idx2.get(tokens2[i], token2idx2['UNK']))
            output_token3.append(token2idx3.get(tokens3[i], token2idx3['UNK']))
    return tokens1, output_token1, output_token2, output_token3, output_label


def code_convert(tokens1, tokens2, tokens3, token2idx1,token2idx2,token2idx3, code_map):
    output_label = []
    output_token1 = []
    output_token2 = []
    output_token3 = []
    for i, token in enumerate(tokens3):
        output_label.append(token2idx3.get(tokens3[i], token2idx3['UNK']))
        output_token1.append(token2idx1.get(tokens1[i], token2idx1['UNK']))
        output_token2.append(token2idx2.get(tokens2[i], token2idx2['UNK']))
        output_token3.append(token2idx3.get(tokens3[i], token2idx3['UNK']))
    return output_token1, output_token2, output_token3, output_label

def seq_padding_multi(tokens, max_len, token2idx=None):
    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if i < token_len:
            seq.append(tokens[i])
        else:
            seq.append(i)
    return seq

def index_seg(tokens, symbol='SEP'):
    flag = 0
    seg = []

    for token in tokens:
        if token == symbol:
            seg.append(flag)
            if flag == 0:
                flag = 1
            else:
                flag = 0
        else:
            seg.append(flag)
    return seg


def position_idx(tokens, symbol='SEP'):
    pos = []
    flag = 0

    for token in tokens:
        pos.append(flag)
        flag += 1

    return pos


def seq_padding(tokens, max_len, token2idx=None, symbol=None):
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                seq.append(token2idx.get(tokens[i]))
            else:
                seq.append(token2idx.get(symbol))
    return seq