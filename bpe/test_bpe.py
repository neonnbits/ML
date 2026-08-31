from main import count_pairs, merge_pair, merge_vocab, encode, decode, _train_on_text

SPECIAL = ["<|endoftext|>"]
# --- merge_pair ---

def test_merge_pair_merges_adjacent():
    assert merge_pair(('l', 'o', 'w'), ('l', 'o')) == ['lo', 'w']

def test_merge_pair_no_match_unchanged():
    assert merge_pair(('a', 'b', 'c'), ('x', 'y')) == ['a', 'b', 'c']

def test_merge_pair_overlapping_not_double_consumed():
    assert merge_pair(('a', 'a', 'a'), ('a', 'a')) == ['aa', 'a']

def test_merge_pair_multiple_occurrences():
    assert merge_pair(('a', 'b', 'a', 'b'), ('a', 'b')) == ['ab', 'ab']

def test_merge_pair_single_symbol():
    assert merge_pair(('a',), ('a', 'a')) == ['a']


# --- count_pairs ---

def test_count_pairs_weights_by_frequency():
    vocab = {('a', 'b', '</w>'): 3}
    assert count_pairs(vocab) == {('a', 'b'): 3, ('b', '</w>'): 3}

def test_count_pairs_sums_across_words():
    vocab = {('a', 'b', '</w>'): 1, ('a', 'b', 'c', '</w>'): 2}
    counts = count_pairs(vocab)
    assert counts[('a', 'b')] == 3

def test_count_pairs_empty_for_single_symbol_words():
    assert count_pairs({('a',): 5}) == {}


# --- merge_vocab ---

def test_merge_vocab_applies_merge_to_all_words():
    vocab = {('l', 'o', 'w', '</w>'): 2, ('l', 'o', '</w>'): 1}
    merged = merge_vocab(vocab, ('l', 'o'))
    assert merged == {('lo', 'w', '</w>'): 2, ('lo', '</w>'): 1}


# --- train ---

def test_train_is_deterministic_and_stops_when_no_pairs_left():
    # "aa" fully merges in 2 steps; asking for 10 must not loop or crash
    vocab, merges = _train_on_text("aa aa", 287, SPECIAL)
    assert merges == [(b'a', b'a'), (b' ', b'aa')]

def test_train_respects_num_merges():
    vocab, merges = _train_on_text("low low lower newest", 260, SPECIAL)
    assert len(merges) == 3


# --- encode / decode ---

def test_roundtrip():
    text = "low low low lower lower newest"
    merges = _train_on_text(text, 5, SPECIAL)
    assert decode(encode(text, merges)) == text

def test_roundtrip_unseen_word():
    # a word not in training data still encodes/decodes correctly
    merges = _train_on_text("low lower", 5, SPECIAL)
    assert decode(encode("slow", merges)) == "slow"
