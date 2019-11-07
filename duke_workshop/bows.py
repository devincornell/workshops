
from sklearn.feature_extraction.text import CountVectorizer

def make_bow_matrix(tokenized_docs, min_df=1, min_doc_wordcount=1):
    vectorizer = CountVectorizer(tokenizer = lambda x: x, preprocessor=lambda x:x, min_df=min_df)
    corpus = vectorizer.fit_transform(tokenized_pars)
    vocab = vectorizer.get_feature_names()
    COOC = corpus.toarray()
    
    # drop docs with less than min_doc_wordcount words in vocab
    zerosel = COOC.sum(axis=1) < min_doc_wordcount
    COOC = COOC[~zerosel]

    return COOC, zerosel