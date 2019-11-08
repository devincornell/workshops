# Intro to Text Analysis Using Spacy and Numpy

I've created this workshop to teach the basics of text analysis in Python with Spacy and Numpy. It will be in workshop format and we can all use a small dataset of US National Security Strategy documents produced by both Trump and Obama. Using SpaCy I'll show how to use tokenization, Named Entity Recognition, Part of Speech Tagging, and grammatical Parse Trees on your text corpora. Then using Sklearn and Numpy matrices I'll show you how to build bag-of-word document matrices, build topic models, calculate pointwise mutual information representations, and build very basic word embedding models. Finally, I'll show a fun example of an "actor network" built from subject-verb-object triplets to compare the two national security strategy documents for their narrative content.

## Notebook Descriptions

### 1_spacy_intro.ipynb
  - tokenization and token information
  - named entity recognition (NER)
  - sentences and part-of-speece (POS)
  - parse trees using prepositional phrase and noun-verb pair extraction
### 2_texts_as_matrices.ipynb
  - basic generalizable tokenization formula
  - bag-of-words (BoW) document term matrix (DTM)
  - word distribution comparisons
  - topic modeling using lda and nmf
  - pointwise mutual information (PMI/PPMI)
  - token co-occurrence matrices
  - singular value decomposition (SVD) word embeddings
### 3_actor_network.ipynb
  - extract subject-verb-object triplets where subject or object are named entities
  - create networkx network with verb relations between entities
  - compare networks between Obama and Trump NSS documents



