# Texts as Matrices


```python
import numpy as np
import spacy
from collections import Counter
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
nlp = spacy.load('en')
```


```python
def read_paragraphs(fname):
    with open(fname, 'r') as f:
        text = f.read()
    paragraphs = [p for p in text.split('\n\n') if len(p) > 0]
    return paragraphs

trump_par_texts = read_paragraphs('nss/trump_nss.txt')
obama_par_texts = read_paragraphs('nss/obama_nss.txt')
par_texts = trump_par_texts + obama_par_texts
k = len(trump_par_texts)
len(par_texts), len(trump_par_texts), len(obama_par_texts)
```




    (550, 400, 150)



## Tokenization Formula
A common step to many text analysis algorithms is to first convert the raw text into sets of tokens. Spacy does most of the work here, there are just a few decisions that need to be made depending on the application: which tokens to include and how to represent the tokens as strings.


```python
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

tokenized_pars = [parse_doc(par) for par in nlp.pipe(par_texts)]
```


```python
# first paragraph, first five tokens
tokenized_pars[0][:5]
```




    ['an', 'America', 'that', 'is', 'safe']



## Bag-of-Words and Document-Term Matrices


```python
min_tf = 5
vectorizer = CountVectorizer(tokenizer = lambda x: x, preprocessor=lambda x:x, min_df=min_tf)
corpus = vectorizer.fit_transform(tokenized_pars)
vocab = vectorizer.get_feature_names()
COOC = corpus.toarray()
len(vocab), type(corpus), corpus.shape, COOC.shape
```




    (1002, scipy.sparse.csr.csr_matrix, (550, 1002), (550, 1002))



Now reduce vocabulary to words which appear at least once in both corpora.


```python
valid_cols = (COOC[:k].sum(axis=0) > 0) & (COOC[k:].sum(axis=0) > 0)
rm_wordids = np.argwhere(~valid_cols)[:,0]
vocab = [w for i,w in enumerate(vocab) if i not in rm_wordids]
COOC = COOC[:,valid_cols]
COOC.shape
```




    (550, 907)



Now remove documents that have none of the selected vocab words.


```python
zerosel = COOC.sum(axis=1)==0
zeroind = np.argwhere(zerosel)[:,0]
tokenized_pars = [toks for i,toks in enumerate(tokenized_pars) if i not in zeroind]
par_texts = [par for i,par in enumerate(par_texts) if i not in zeroind]
k = k - (zeroind < k).sum()
COOC = COOC[~zerosel]

COOC.shape, k, len(tokenized_pars), len(par_texts)
```




    ((534, 907), 391, 534, 534)




```python
print(vocab[:10])
corpus[:10,:10].toarray()
```

    ['(', ')', ',', '-', '.', ':', ';', 'Afghanistan', 'Africa', 'African']





    array([[0, 0, 6, 0, 3, 0, 0, 0, 0, 0],
           [0, 0, 3, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 6, 1, 4, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 8, 1, 4, 0, 0, 0, 0, 0],
           [0, 0, 4, 0, 2, 0, 0, 0, 0, 0],
           [0, 0, 5, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 3, 0, 3, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 2, 0, 0, 0, 0, 0]])



Using bag of words, we now compare word distributions averaged across the documents from Trump and Obama. Here we present the words that are more likely to be in the Trump NSS compared to Obama's.


```python
topn = 30
trump_cts, obama_cts = COOC[:k].sum(axis=0), COOC[k:].sum(axis=0)
logdiff = np.log( (trump_cts/trump_cts.sum()) / (obama_cts / obama_cts.sum()))
indices = logdiff.argsort()[-topn:][::-1]
print(', '.join(['{} ({:.2f})'.format(vocab[idx],logdiff[idx]) for idx in indices]))
```

    : (3.05), priority (2.56), missile (2.28), compete (2.28), under (2.21), conditions (2.13), immigration (2.13), nation (2.09), industry (2.04), seeks (2.04), liberty (2.04), Americans (2.01), continues (1.95), sovereign (1.95), minded (1.95), adversaries (1.91), life (1.90), encourage (1.88), intellectual (1.84), base (1.84), want (1.84), was (1.84), communications (1.84), identify (1.84), deterrence (1.84), principles (1.72), understand (1.72), criminals (1.72), actions (1.70), technologies (1.68)


### Topic Modeling
Because topic models are computed directly from document-term matrices, I demonstrate the use of both the NMF and LDA algorithms. After computing each model, I then compute the log ratio of probabilities of subcorpora being associated with each topic. Larger values mean Trump's documents are more closely associated with the topic while more negative values are more closely associated with Obama.


```python
# non-negative matrix factorization (similar to pca but for only positive-entry matrices)
nmf_model = NMF(n_components=10).fit(COOC)
doc_topics = nmf_model.transform(COOC)
topic_words = nmf_model.components_
topic_words.shape, doc_topics.shape
```




    ((10, 907), (534, 10))




```python
# for nmf compare distributions between sources
trump_av = doc_topics[:k].mean(axis=0)
obama_av = doc_topics[k:].mean(axis=0)
logratio = np.log(trump_av/obama_av)
logratio
```




    array([-0.40763258, -0.60503723, -0.83237009, -0.74170717, -0.42066011,
           -0.4556857 , -0.86135373, -0.14774911, -1.7632974 , -0.41005808])




```python
# non-negative matrix factorization (similar to pca but for only positive-entry matrices)
lda_model = LatentDirichletAllocation(n_components=10).fit(COOC)
doc_topics = lda_model.transform(COOC)
topic_words = lda_model.components_
topic_words.shape, doc_topics.shape
```




    ((10, 907), (534, 10))




```python
# for nmf compare distributions between sources
trump_av = doc_topics[:k].mean(axis=0)
obama_av = doc_topics[k:].mean(axis=0)
logratio = np.log(trump_av/obama_av)
logratio
```




    array([-0.55882016,  2.0369791 ,  0.7375232 ,  0.04177753,  1.32832417,
           -0.47229356,  0.0924281 , -0.11757099, -0.64856747,  0.32114886])



### Pointwise Mutual Information
The [pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information) calculates the level of association between two variables, document and word in this case (as designated by the 2-dimensional distribution), by controlling for both the frequency of a word and the number of words in a document. Higher values mean that the word is more uniquely associated with the document statistically than not.

Positive pointwise mutual information is a variant of PMI which sets negative values (words which are less associated with documents than expected) to zero. While we loose some information here, this solves the problem of -infinity values caused by taking log(0) and has shown to still be a robust measure.
Levy, Goldberg, Dagan (2015) _Improving Distributional Similarity with Lessons Learned from Word Embeddings_ ([link])(https://levyomer.files.wordpress.com/2015/03/improving-distributional-similarity-tacl-2015.pdf).


```python
import matrices # from included script
```


```python
PPMI = matrices.calc_ppmi(COOC)
print(PPMI.shape)
PPMI[:5,:5]
```

    (534, 907)





    array([[0.        , 0.        , 6.40856958, 0.        , 5.99292046],
           [0.        , 0.        , 6.33415782, 0.        , 5.61440998],
           [0.        , 0.        , 0.        , 0.        , 7.39511952],
           [0.        , 0.        , 6.16476272, 6.76932358, 5.96487521],
           [0.        , 0.        , 0.        , 0.        , 6.98965443]])



The power of PMI is that you can go back to the original documents to examine the words most closely associated with them compared to all other docs in the corpus.


```python
target_docid = 0
temp_PPMI = PPMI.copy()
for i in range(5):
    idx = temp_PPMI[target_docid,:].argmax()
    print('{} ({:.2f})'.format(vocab[idx], temp_PPMI[target_docid,idx]), end=' ')
    temp_PPMI[target_docid,idx] = 0
print('\n', par_texts[target_docid])
```

    uphold (10.04) foundation (9.96) liberty (9.71) confidence (9.62) safe (9.60) 
     An America that is safe, prosperous, and free at home is an America with the strength, confidence, and will to lead abroad. It is an America that can preserve peace, uphold liberty , and create enduring advantages for the American people. Putting America first is the duty of our government and the foundation for U.S. leadership in the world.


### Singular Value Decomposition of PPMI Matrix


```python
SVD = matrices.calc_svd(PPMI,10)
svd_vars = SVD.var(axis=0)/SVD.var(axis=0).sum()
svd_vars # variance explained by each dimension (if arg two of calc_svd is None), need to normalize by all eigenvalues
```




    array([0.26943374, 0.11470561, 0.09687489, 0.09364945, 0.0834587 ,
           0.08070123, 0.07893944, 0.06435028, 0.06017203, 0.05771463])



We can now examine these SVD representations as vectors in an embedding space: these are essentially document (paragraph in this case) embeddings. Now we can examine how different documents are distributed across the space.


```python
# calcualte average veector norm and norm of averagae vector
np.linalg.norm(SVD, axis=1).mean(), np.linalg.norm(SVD.mean(axis=0))
```




    (23.912579290551182, 19.19038751067269)



Because we have document embeddings, we can measure the field that situates each document. In this case, we'll identify the document closest to the mean vector (center of the field), and then the document furthest from the mean.


```python
dists = np.linalg.norm(SVD - SVD.mean(axis=0), axis=1)
docind = dists.argsort()[::-1]
```


```python

```

## Token Co-Occurrence Matrices


```python
trump_cooc = (corpus[:k].T*corpus[:k]).toarray()
obama_cooc = (corpus[k:].T*corpus[k:]).toarray()
trump_cooc.shape, obama_cooc.shape
```




    ((907, 907), (907, 907))




```python

```




    numpy.ndarray




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
# compute vocab from words that appear at least once in each subcorpora
min_count = 1
trump_tok_cts = Counter([tok for doc in trump_pars for tok in doc])
obama_tok_cts = Counter([tok for doc in obama_pars for tok in doc])
words = set(trump_tok_cts.keys()) & set(obama_tok_cts.keys())
word_cts = {w:(obama_tok_cts[w]+trump_tok_cts[w]) for w in words}
vocab = list(sorted(word_cts.items(), key=lambda x: x[0], reverse=True))
n = len(vocab)
n
```

## Word Co-Occurrence Matrix
One common technique used in corpus linguistics and other social sciences is the co-occurrence matrix. This model of the text is implicitly used in word2vec algorithms. It can be defined in a number of ways, but here we will define it as the number of times two words appear in the same paragraph together.


```python
def calc_cooccur(docs):
    cooccur = dict()
    for doc in docs:
        for i,w1 in enumerate(doc):
            for j,w2 in enumerate(doc[i:]):
                k = (w1,w2)
                if k not in cooccur:
                    cooccur[k] = 0
                cooccur[k] += 1
    return cooccur
trump_cooc_dict = calc_cooccur(trump_pars)
obama_cooc_dict = calc_cooccur(obama_pars)
print(list(trump_cooc_dict.keys())[:4])
```

    [('an', 'an'), ('an', 'America'), ('an', 'that'), ('an', 'is')]



```python
def cooc_matrix(cooc_dict, vocab):
    voc_index = {tok:i for i,tok in enumerate(vocab)}
    cooc_mat = np.zeros((n,n))
    for toks,ct in cooc_dict.items():
        w1,w2 = toks
        cooc_mat[voc_index[w1],voc_index[w2]] = ct
    return cooc_mat
trump_cooc = cooc_matrix(trump_cooc_dict, vocab)
obama_cooc = cooc_matrix(trump_cooc_dict, vocab)
```


```python

```
