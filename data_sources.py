from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import gutenberg
from nltk.corpus import brown
import os.path

def get_newsgroup_data(useN=100):
    
    nd = fetch_20newsgroups(shuffle=True, random_state=0)
    texts, fnames =  nd['data'][:useN], nd['filenames'][:useN]
    fnames = [fn.split('/')[-1] for fn in fnames]
    
    return texts, fnames

def get_gutenberg_data(useN=100):
    try:
        fileids = gutenberg.fileids()
    except LookupError:
        import nltk
        nltk.download('gutenberg')
        fileids = gutenberg.fileids()
        
    fileids = fileids[:useN]
    texts = [gutenberg.raw(fid) for fid in fileids]
    
    fileids = [os.path.splitext(fid)[0] for fid in fileids]
    
    return texts, fileids



def get_brown_data(useN=100):
    try:
        fileids = brown.fileids()
    except LookupError:
        import nltk
        nltk.download('brown')
        fileids = brown.fileids()
        
    fileids = fileids[:useN]
    texts = [brown.raw(fid) for fid in fileids]
    
    fileids = [os.path.splitext(fid)[0] for fid in fileids]
    
    return texts, fileids

