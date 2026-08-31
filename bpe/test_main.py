import unicodedata

from main import train, encode, decode, _train_on_text

SPECIAL = ["<|endoftext|>"]

def test_create_file(tmp_path):
    d = tmp_path / "corpus.txt"
    d.write_text("low lower lowest newest")
    vocab, merges = train(d, 267, SPECIAL)
    assert len(vocab) == 267

def roundtrip(text, num_merges=277):
    vocab, merges = _train_on_text(text, num_merges, SPECIAL)
    tokens = encode(text, merges)
    return decode(tokens), tokens


# --- 1. exact round-trip: whitespace is data, not noise -----------------

def test_roundtrip_tabs():
    text = "col1\tcol2\tcol3\n1\t2\t3"
    out, _ = roundtrip(text)
    assert out == text


def test_roundtrip_double_spaces():
    text = "two  spaces   three    four"
    out, _ = roundtrip(text)
    assert out == text
    assert "  " in out  # not collapsed


def test_roundtrip_newlines():
    text = "line one\nline two\n\n\nafter blank lines\n"
    out, _ = roundtrip(text)
    assert out == text
    assert out.endswith("\n")  # trailing newline survives (old .rstrip() would kill it)


def test_roundtrip_leading_trailing_whitespace():
    text = "   padded   "
    out, _ = roundtrip(text)
    assert out == text


def test_roundtrip_mixed_whitespace():
    text = " \t \n\t mixed \r\n\t"
    out, _ = roundtrip(text)
    assert out == text


def test_roundtrip_zero_merges():
    # with no merges every token is a single byte; must still reconstruct exactly
    text = "hello\tworld\n"
    vocab, merges = _train_on_text(text, 0, SPECIAL)
    assert merges == []
    tokens = encode(text, merges)
    assert all(len(t) == 1 for t in tokens)
    assert len(tokens) == len(text.encode("utf-8"))
    assert decode(tokens) == text


# --- 2. compression: merges actually shorten the token stream -------------

def test_compression():
    text = "low low low low lower lower lowest newest newest widest " * 20
    n_bytes = len(text.encode("utf-8"))

    vocab, merges = _train_on_text(text, 287, SPECIAL)
    tokens = encode(text, merges)

    assert len(tokens) < n_bytes
    # repeated words should collapse to very few tokens per word
    assert len(tokens) < n_bytes / 2
    assert decode(tokens) == text


def test_more_merges_never_longer():
    text = "the cat sat on the mat the cat sat on the hat " * 10
    lengths = []
    for n in (0, 5, 10, 20, 40):
        vocab, merges = _train_on_text(text, n, SPECIAL)
        lengths.append(len(encode(text, merges)))
    assert lengths == sorted(lengths, reverse=True)


# --- 3. multibyte: byte-level BPE must never corrupt UTF-8 ----------------

def test_roundtrip_cjk():
    text = "你好，世界。今日は良い天気ですね。안녕하세요"
    out, tokens = roundtrip(text)
    assert out == text
    assert all(isinstance(t, bytes) for t in tokens)


def test_roundtrip_nfd_hangul():
    # NFD decomposes precomposed syllables into jamo sequences; must survive untouched
    text = unicodedata.normalize("NFD", "한글 테스트")
    assert text != "한글 테스트"  # sanity: decomposition happened
    out, _ = roundtrip(text)
    assert out == text
    assert unicodedata.normalize("NFD", out) == out  # still NFD, not re-normalized


def test_roundtrip_zwj_emoji():
    family = "👨‍👩‍👧‍👦"  # ZWJ sequence
    flag = "🇯🇵"  # regional indicator pair
    skin = "👍🏽"  # base + skin tone modifier
    text = f"family: {family} flag: {flag} thumbs: {skin}"
    out, _ = roundtrip(text)
    assert out == text


def test_multibyte_merges_form_whole_chars():
    # _train_on_text on repeated CJK so merges form; the merged tokens should decode cleanly
    text = "你好你好你好你好你好"
    vocab, merges = _train_on_text(text, 267, SPECIAL)
    tokens = encode(text, merges)
    assert decode(tokens) == text
    assert len(tokens) < len(text.encode("utf-8"))


# --- special tokens -------------------------------------------------------

def test_special_token_never_merged_in_training():
    text = "aaa<|endoftext|>aaa<|endoftext|>aaa"
    vocab, merges = _train_on_text(text, 267, SPECIAL)
    for a, b in merges:
        assert b"<" not in a + b
        assert b"|" not in a + b
        assert b"endoftext" not in a + b
