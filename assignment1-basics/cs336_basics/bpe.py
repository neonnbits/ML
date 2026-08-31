import collections
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(text):
    ptoken_counter = collections.Counter(re.findall(PAT, text))
    return ptoken_counter
        
def count_pairs(id_to_ptok, id_to_freq):
    pair_counts = {}
    pair_to_ptok = {}
    for index, ptok in id_to_ptok.items():
        for i in range(len(ptok)-1):
            pair = (ptok[i], ptok[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + id_to_freq[index] 
            pair_to_ptok.setdefault(pair, set()).add(index)
    
    return pair_counts, pair_to_ptok

def merge_vocab(pair_counts, pair_to_ptok, id_to_ptok, id_to_freq, best_pair):
    ptok_ids = list(pair_to_ptok[best_pair])
    for ind in ptok_ids:
        word = id_to_ptok[ind]
        new_word = merge_pair(word, best_pair)
        # subtract pairs
        for i in range(len(word)-1):
            pair = (word[i], word[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) - id_to_freq[ind] 
            pair_to_ptok.setdefault(pair, set()).discard(ind)
        #add pairs
        for i in range(len(new_word)-1):
            pair = (new_word[i], new_word[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + id_to_freq[ind] 
            pair_to_ptok.setdefault(pair, set()).add(ind)

        id_to_ptok[ind] = new_word


def merge_pair(word, best_pair):
    new_word = []
    i=0
    while(i<len(word)-1):
        pair = (word[i], word[i+1])
        if((pair) == best_pair):
            new_word.append(word[i]+word[i+1])
            i+=2
        else:
            new_word.append(word[i])
            i+=1
    if(i<len(word)):
        new_word.append(word[i])

    return new_word


def encode(text, merges):
    symbols = []
    for token in re.finditer(PAT, text):
        symbols.append(tuple(bytes([b]) for b in token.group().encode("utf-8")))

    for merge in merges:
        for i in range(len(symbols)):
            symbols[i] = merge_pair(symbols[i], merge)

    return [t for word in symbols for t in word]

def decode(tokens):
    return b"".join(tokens).decode("utf-8", errors='replace')

def split_on_special(text, special_tokens):
    if not special_tokens:
        return [text]
    pattern = "|".join(re.escape(t) for t in special_tokens)
    return re.split(pattern, text)

def _train_on_text(text, vocab_size, special_tokens):
    docs = split_on_special(text, special_tokens)
    doc_counter = collections.Counter()
    for doc in docs:
        doc_counter.update(pretokenize(doc))

    id_to_ptok = {}
    id_to_freq = {}
    for i, (word, freq) in enumerate(doc_counter.items()):
        pbytes = tuple(bytes([b]) for b in word.encode("utf-8"))
        id_to_ptok[i] = pbytes
        id_to_freq[i] = freq
    
    merges = []
    merge_count = vocab_size - 256 - len(special_tokens)

    pair_counts, pair_to_ptok = count_pairs(id_to_ptok, id_to_freq)
    for _ in range(merge_count):
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        if pair_counts[best_pair] == 0:
            break
        merges.append(best_pair)
        merge_vocab(pair_counts, pair_to_ptok, id_to_ptok, id_to_freq, best_pair)
        assert pair_counts[best_pair] == 0
        del pair_counts[best_pair]

    return merges

def train(file_path, vocab_size, special_tokens):
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
    merges = _train_on_text(text, vocab_size, special_tokens)

    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i, spl_token in enumerate(special_tokens, start=256):
        vocab[i] = spl_token.encode("utf-8")

    offset = len(vocab)
    for i, (a,b) in enumerate(merges, start=offset):
        vocab[i] = a+b

    return vocab, merges