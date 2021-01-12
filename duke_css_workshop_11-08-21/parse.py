
from sklearn.feature_extraction.text import CountVectorizer

def read_paragraphs(fname):
    with open(fname, 'r') as f:
        text = f.read()
    paragraphs = [p for p in text.split('\n\n') if len(p) > 0]
    return paragraphs


def parse_tok(tok):
    '''Convert spacy token object to string.'''
    number_ents = ('NUMBER','MONEY','PERCENT','QUANTITY','CARDINAL','ORDINAL')
    if tok.ent_type_ == '':
        return tok.text.lower()
    elif tok.ent_type_ in number_ents:
        return tok.ent_type_
    else:
        return tok.text
    
def use_tok(tok):
    '''Decide to use token or not.'''
    return tok.is_ascii and not tok.is_space and len(tok.text.strip()) > 0
    
def parse_doc(doc):
    # combine multi-word entities into their own tokens (just a formula)
    for ent in doc.ents:
        ent.merge(tag=ent.root.tag_, ent_type=ent.root.ent_type_)
    return [parse_tok(tok) for tok in doc if use_tok(tok)]


def make_bow_matrix(tokenized_docs, min_df=1, min_doc_wordcount=1):
    vectorizer = CountVectorizer(tokenizer = lambda x: x, preprocessor=lambda x:x, min_df=min_df)
    corpus = vectorizer.fit_transform(tokenized_docs)
    vocab = vectorizer.get_feature_names()
    COOC = corpus.toarray()
    
    # drop docs with less than min_doc_wordcount words in vocab
    zerosel = COOC.sum(axis=1) < min_doc_wordcount
    COOC = COOC[~zerosel]

    return COOC, vocab, zerosel
