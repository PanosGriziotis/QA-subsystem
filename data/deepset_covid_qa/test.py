from sacremoses import MosesTokenizer, MosesDetokenizer
from simalign import SentenceAligner 
from collections import defaultdict


def compute_alignment(source_sentence , translated_sentence):
    source_sentence = tokenize(source_sentence, lang='en')
    print (source_sentence)
    translated_sentence = tokenize(translated_sentence, lang='el')

    # alignments with simalign
    myaligner = SentenceAligner(model="bert-base-multilingual-cased", token_type="bpe", matching_methods="mai", device="gpu")
    alignment = myaligner.get_word_aligns (source_sentence, translated_sentence)
    #alignment = ' '.join([str(pair[0]) + '-' + str(pair[1]) for pair in alignment])
    
    return alignment


# PROCESSING TEXT
tokenizer_en = MosesTokenizer(lang='en')
detokenizer_en = MosesDetokenizer(lang='en')
tokenizer_el = MosesTokenizer(lang='el')
detokenizer_el = MosesDetokenizer(lang='el')

MAX_NUM_TOKENS = 10
SPLIT_DELIMITER = ';'
LANGUAGE_ISO_MAP = {'en': 'english', 'el': 'greek'}


def tokenize(text, lang, return_str=True):
    if lang == 'en':
        text_tok = tokenizer_en.tokenize(text, return_str=return_str, escape=False)
        return text_tok
    elif lang == 'el':
        text_tok = tokenizer_el.tokenize(text, return_str=return_str, escape=False)
        return text_tok

def tok2char_map(text_raw, text_tok):
    # First, compute the token to white-spaced token indexes map (many-to-one map)
    # tok --> ws_tok
    tok2ws_tok = dict()
    ws_tokens = text_raw.split()
    idx_wst = 0
    merge_tok = ''
    for idx_t, t in enumerate(text_tok.split()):
        merge_tok += t
        tok2ws_tok[idx_t] = idx_wst
        if merge_tok == ws_tokens[idx_wst]:
            idx_wst += 1
            merge_tok = ''

    # Second, compute white-spaced token to character indexes map (one-to-one map):
    # ws_tok  --> char
    ws_tok2char = dict()
    for ws_tok_idx, ws_tok in enumerate(text_raw.split()):
        if ws_tok_idx == 0:
            char_idx = 0
            ws_tok2char[ws_tok_idx] = char_idx
        elif ws_tok_idx > 0:
            char_idx = len(' '.join(text_raw.split()[:ws_tok_idx])) + 1
            ws_tok2char[ws_tok_idx] = char_idx

    # Finally, compute the token to character map (one-to-one)
    tok2char = {tok_idx: ws_tok2char[tok2ws_tok[tok_idx]]
                for tok_idx, _ in enumerate(text_tok.split())}

    return tok2char

def get_src2tran_alignment_char(alignment, source, translation):
    source_tok = tokenize(source, 'en') 
    translation_tok = tokenize(translation, 'el')
    src_tok2char = tok2char_map(source, source_tok)
    tran_tok2char = tok2char_map(translation, translation_tok)

    # Get token index to char index translation map for both source and target
    src2tran_alignment_char = defaultdict(list)
    # Prevent
    try:
        for src_tran in alignment.split():
            src_tok_idx = int(src_tran.split('-')[0])
            tran_tok_idx = int(src_tran.split('-')[1])
            src_char_idx = src_tok2char[src_tok_idx]
            tran_char_idx = tran_tok2char[tran_tok_idx]
            src2tran_alignment_char[src_char_idx].append(tran_char_idx)
    except KeyError:
        pass
    # Define a one-to-one mapping left-oriented by keeping the minimum key value
    src2tran_alignment_char_min_tran_index = {k: min(v) for k, v in src2tran_alignment_char.items()}

    return src2tran_alignment_char_min_tran_index

# Check is the content of SQUAD has been translated and aligned already
