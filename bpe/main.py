def count_pairs(vocab):
    pair_counts = {}

    for word, freq in vocab.items():
        for i in range(len(word)-1):
            pair = (word[i], word[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq 
    
    return pair_counts

def merge_vocab(vocab, best_pair):
    new_vocab = {}
    for word, freq in vocab.items():
        new_word = merge_pair(word, best_pair)
        new_vocab[tuple(new_word)] = freq

    return new_vocab

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
    words = text.split()
    symbols = []
    for word in words:
        token = tuple(word) + ('</w>',)
        symbols.append(token)

    for merge in merges:
        for i in range(len(symbols)):
            symbols[i] = merge_pair(symbols[i], merge)

    return [t for word in symbols for t in word]

def decode(tokens):
    return "".join(tokens).replace('</w>', ' ').rstrip()

def train(text, num_merges):
    words = text.split()
    vocab = {}
    for word in words:
        symbols = tuple(word) + ('</w>',)
        vocab[symbols] = vocab.get(symbols, 0)+1

    merges = []

    for _ in range(num_merges):
        pair_counts = count_pairs(vocab)
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        vocab = merge_vocab(vocab, best_pair)

    return merges
if __name__ == "__main__":
    text = "low low low lower lower newest"
    num_merges = 5
    merges = train(text, num_merges)
    enc = encode(text, merges)
    print(decode(enc))