# Spacy Basics
Devin J. Cornell, Nov 2019

To begin:


## Read Text File, Convert to Paragraphs


```python
with open('nss/trump_nss.txt', 'r') as f:
    text = f.read()
print(text[:500], '...')
```

    An America that is safe, prosperous, and free at home is an America with the strength, confidence, and will to lead abroad. It is an America that can preserve peace, uphold liberty , and create enduring advantages for the American people. Putting America first is the duty of our government and the foundation for U.S. leadership in the world.
    
    A strong America is in the vital interests of not only the American people, but also those around the world who want to partner with the United States in p ...



```python
paragraphs = [p for p in text.split('\n\n') if len(p) > 0]
type(paragraphs), type(paragraphs[0]), len(paragraphs)
# each paragraph is a string in the list
```




    (list, str, 400)



## Load Spacy
For more language models see [Spacy Documentation on Models & Languages](https://spacy.io/usage/models)


```python
import spacy # import the package
nlp = spacy.load('en') # load a language model
nlp
```




    <spacy.lang.en.English at 0x7f9414349dd8>



## Work With Spacy Doc Object
A spacy doc object is returned after parsing a document using `nlp()`. You can see the object properties in the [spacy Doc object documentation](https://spacy.io/api/doc).


```python
# here the first paragraph is parsed with spacy and the result is in the variable "doc"
doc = nlp(paragraphs[0])
type(doc)
```




    spacy.tokens.doc.Doc



You can treat doc as an iterator across [Token objects](https://spacy.io/api/token). See a list of token attributes in the [spacy token attributes documentation](https://spacy.io/api/token#attributes).



```python
print(type(doc[0]))
for tok in doc[:3]:
    print(tok)
```

    <class 'spacy.tokens.token.Token'>
    An
    America
    that


### Regular Token Information


```python
for tok in doc[:5]:
    print('{}: '
          'lower-case string: "{}", '
          'whitespace: "{}", '
          'string with whitespace: "{}", '
          'lemma: "{}", '
          'prefix: "{}", '
          'suffix: "{}"'
          '\n'.format(tok.text, tok.lower_, tok.whitespace_, tok.text_with_ws, tok.lemma_, tok.prefix_, tok.suffix_))
```

    An: lower-case string: "an", whitespace: " ", string with whitespace: "An ", lemma: "an", prefix: "A", suffix: "An"
    
    America: lower-case string: "america", whitespace: " ", string with whitespace: "America ", lemma: "America", prefix: "A", suffix: "ica"
    
    that: lower-case string: "that", whitespace: " ", string with whitespace: "that ", lemma: "that", prefix: "t", suffix: "hat"
    
    is: lower-case string: "is", whitespace: " ", string with whitespace: "is ", lemma: "be", prefix: "i", suffix: "is"
    
    safe: lower-case string: "safe", whitespace: "", string with whitespace: "safe", lemma: "safe", prefix: "s", suffix: "afe"
    



```python
# can put doc back into string by using text_with_ws
''.join([tok.text_with_ws for tok in doc])
```




    'An America that is safe, prosperous, and free at home is an America with the strength, confidence, and will to lead abroad. It is an America that can preserve peace, uphold liberty , and create enduring advantages for the American people. Putting America first is the duty of our government and the foundation for U.S. leadership in the world.'




```python
# this shows a few token flags assigned to tokens.
for tok in doc:
    if tok.is_punct:
        print('is_punct:', tok)
    elif tok.is_digit:
        print('is_digit:', tok)
    elif tok.is_upper:
        print('is_upper:', tok)
    elif tok.is_stop:
        print('is_stop:', tok)
```

    is_stop: An
    is_stop: that
    is_stop: is
    is_punct: ,
    is_punct: ,
    is_stop: and
    is_stop: at
    is_stop: is
    is_stop: an
    is_stop: with
    is_stop: the
    is_punct: ,
    is_punct: ,
    is_stop: and
    is_stop: will
    is_stop: to
    is_punct: .
    is_stop: It
    is_stop: is
    is_stop: an
    is_stop: that
    is_stop: can
    is_punct: ,
    is_punct: ,
    is_stop: and
    is_stop: for
    is_stop: the
    is_punct: .
    is_stop: first
    is_stop: is
    is_stop: the
    is_stop: of
    is_stop: our
    is_stop: and
    is_stop: the
    is_stop: for
    is_upper: U.S.
    is_stop: in
    is_stop: the
    is_punct: .


### Named Entity Recognition
You can extract named entity information by using the `tok.ent_type_` attribute. You can see a full list of named entity types in the [spacy named entity documentation](https://spacy.io/api/annotation#named-entities).


```python
# Notice how tok._ent_type_ == '' for cases where the token is not an entity
for tok in doc[:5]:
    print('{}, type: {}'.format(tok, tok.ent_type_))
```

    An, type: 
    America, type: GPE
    that, type: 
    is, type: 
    safe, type: 



```python
# now only print entities
for tok in doc:
    if tok.ent_type_ != '':
        print('{}, type: {}'.format(tok, tok.ent_type_))
```

    America, type: GPE
    America, type: GPE
    America, type: GPE
    American, type: NORP
    America, type: GPE
    first, type: ORDINAL
    U.S., type: GPE


## Sentences and Parse Trees
Spacy also has the ability to parse sentences for gramattical structures. This includes breaking tokens into groups of tokens consituting sentences, assigning Part-of-Speech (POS) tags to each word, and building a Dependency Parse Tree for detailed gramattical structure. For information on parse trees, see the [dependency type list](https://spacy.io/api/annotation#dependency-parsing) and see the [displacy visualization](https://explosion.ai/demos/displacy) for an example.


```python
# just print each sentence
for sent in doc.sents:
    print(sent)
```

    An America that is safe, prosperous, and free at home is an America with the strength, confidence, and will to lead abroad.
    It is an America that can preserve peace, uphold liberty , and create enduring advantages for the American people.
    Putting America first is the duty of our government and the foundation for U.S. leadership in the world.



```python
# extract part of speech from each spacy token
for sent in doc.sents:
    print(' '.join(['{} ({})'.format(tok,tok.pos_) for tok in sent]))
    break
```

    An (DET) America (PROPN) that (PRON) is (AUX) safe (ADJ) , (PUNCT) prosperous (ADJ) , (PUNCT) and (CCONJ) free (ADJ) at (ADP) home (NOUN) is (AUX) an (DET) America (PROPN) with (ADP) the (DET) strength (NOUN) , (PUNCT) confidence (NOUN) , (PUNCT) and (CCONJ) will (AUX) to (PART) lead (VERB) abroad (ADV) . (PUNCT)



```python
# there is also a more fine-grained version
for sent in doc.sents:
    print(' '.join(['{} ({})'.format(tok,tok.tag_) for tok in sent]))
    break
```

    An (DT) America (NNP) that (WDT) is (VBZ) safe (JJ) , (,) prosperous (JJ) , (,) and (CC) free (JJ) at (IN) home (NN) is (VBZ) an (DT) America (NNP) with (IN) the (DT) strength (NN) , (,) confidence (NN) , (,) and (CC) will (MD) to (TO) lead (VB) abroad (RB) . (.)


### Parse Tree Examples



```python
# get all prepositional phrases
def prep_phrases(doc):
    phrases = list()
    for tok in doc:
        if tok.pos_ == 'ADP':
            pp = ''.join([t.orth_ + t.whitespace_ for t in tok.subtree])
            phrases.append(pp)
    return phrases
prep_phrases(doc)
```




    ['at home ',
     'with the strength, confidence',
     'for the American people',
     'of our government and the foundation for U.S. leadership in the world',
     'for U.S. leadership in the world',
     'in the world']




```python
def noun_verb_pairs(doc):
    nounverbs = list()
    for tok in doc:
        if tok.dep_ == 'ROOT':
            nounverbs.append((child(tok,'nsubj'),tok,child(tok,'dobj')))
    return nounverbs

def child(tok, dep): # helper function
    for c in tok.children:
        if c.dep_== dep:
            return c
    return None

noun_verb_pairs(doc)
```




    [(America, is, None), (It, is, None), (None, is, None)]



## Parse Multiple Documents
Here I show several of the previous examples that have been applied to the entire set of documents instead of just one.


```python
from collections import Counter
```


```python
# perform spacy parsing
parsed_pars = list(nlp.pipe(paragraphs))
len(parsed_pars), type(parsed_pars), type(parsed_pars[0])
```




    (400, list, spacy.tokens.doc.Doc)




```python
# get most frequent words
tokens = [tok.text for doc in parsed_pars for tok in doc]
tok_cts = Counter(tokens)
sort_cts = list(sorted(tok_cts.items(), key=lambda x: x[1], reverse=True))
print(sort_cts[:10])
```

    [('and', 1373), (',', 1227), ('.', 995), ('the', 923), ('to', 793), ('of', 582), ('our', 381), ('will', 334), ('that', 269), ('in', 268)]



```python
# get all entities from all documents, count their frequency of use
all_entities = [tok.text for doc in parsed_pars for tok in doc if tok.ent_type_!='']
ent_cts = Counter(all_entities)
sort_cts = list(sorted(ent_cts.items(), key=lambda x: x[1], reverse=True))
print(sort_cts[:10])
```

    [('United', 223), ('States', 219), ('the', 182), ('U.S.', 124), ('American', 121), ('The', 116), ('America', 85), ('China', 33), ('Americans', 32), ('Russia', 23)]



```python
# most common noun-verb pairs
all_nv = [(n.text,v.text) for doc in parsed_pars for n,v,s in noun_verb_pairs(doc) if n is not None]
nv_cts = Counter(all_nv)
sort_cts = list(sorted(nv_cts.items(), key=lambda x: x[1], reverse=True))
print(sort_cts[:5])
```

    [(('We', 'work'), 18), (('We', 'encourage'), 9), (('We', 'support'), 9), (('We', 'are'), 8), (('States', 'continue'), 8)]



```python

```
