import numpy as np

def document_embeddings(X, ndim, alpha=0.75, k=None, check_matrix=True):
	
	if check_matrix:
		if (X.sum(axis=0)==0).sum() > 0:
			raise ValueError('Some columns of X are all zeros.')
		if (X.sum(axis=1)==0).sum() > 0:
			raise ValueError('Some rows of X are all zeros.')
	
	PPMI = calc_ppmi(X, alpha=alpha, k=k)
	SVD = calc_svd(PPMI, ndim)
	SVD_cent = SVD - SVD.mean(axis=0)
	
	return SVD


def calc_ppmi(C, alpha=0.75, k=None):
    '''Calculate PPMI of co-occurrence matrix X.
    Description:
        The implementation of this function is based on
            section 2.1 of [1] (see citation at top).
    Args:
        C (ndarray<n x n>): co-occurrence matrix (smoothed
            or not is okay). Each row is a word's context.
        alpha (float or None): context distribution
            smoothing factor (see "Context Distribution 
            Smoothing" in section 3.2 of [1])
        k (float or None): this parameter is analogous
            to negative sampling in skip-gram models.
            See "Shifted PMI" in section 3.2 of [1] for
            details.
    Returns:
        ndarray<n x n>: ppmi matrix
    
    '''
    # see equation in section 2.1 of "Improving Distributional 
    # Similarity with Lessons Learned from Word Embeddings" by
    # Levy, Goldberg, Dagan (2015)
    
    if alpha is not None:
        # equation (3) in [1] (pg. 215)
        rowsums = C.sum(axis=1)**alpha
        X = (C**alpha) / rowsums[:, np.newaxis]
    else:
        X = C
        
    # marginal word counts
    cx, cy = X.sum(axis=0), X.sum(axis=1)
    cxcy = np.outer(cy,cx).astype(np.float32)
    
    # divide for p(x,y) / ( p(x)p(y) )
    # expression for PMI in section 2 of [1]
    cxcy[cxcy==0] = np.nan
    D = X.shape[0] * X.shape[1]
    pxy = X/cxcy*D
    
    # take log (avoiding negative inf issue)
    pxy[pxy == 0] = np.nan
    pmi = np.log(pxy)
    
    if k is not None:
        # (ssee param k description in docstring)
        pmi = pmi - np.log(k)
        
    # convert pmi -> ppmi by taking all nan and negative 
    #     values to 0.
    ppmi = pmi
    ppmi[np.isnan(pmi)] = 0
    ppmi[pmi < 0] = 0
    #ppmi[~np.isfinite(pmi)] = 0

    return ppmi

def calc_svd(X, n_dim):
    U, sigma, Vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
    X_pca = np.dot(U,np.diag(sigma))[:,:n_dim]
    return X_pca


# JUST CARRIED OVER FROM OLD NOTEBOOK
def max_word_pmi():
    target_docid = 0
    temp_PPMI = PPMI.copy()
    for i in range(5):
        idx = temp_PPMI[target_docid,:].argmax()
        print('{} ({:.2f})'.format(vocab[idx], temp_PPMI[target_docid,idx]), end=' ')
        temp_PPMI[target_docid,idx] = 0
    print('\n', par_texts[target_docid])
    
    
# JUST CARRIED OVER FROM OLD NOTEBOOK
def test_svd():
    SVD = matrices.calc_svd(PPMI,100)
    SVD = SVD - SVD.mean(axis=0) # center
    SVD = SVD / np.linalg.norm(SVD, axis=1)[:,np.newaxis] # normalize
    svd_vars = SVD.var(axis=0)/SVD.var(axis=0).sum()
    svd_vars # variance explained by each dimension (if arg two of calc_svd is None), need to normalize by all eigenvalues

    
def field_center_periphery():
    #dists = np.linalg.norm(SVD - SVD.mean(axis=0), axis=1)
    #dists = SVD.dot(SVD.mean(axis=0))
    dists = SVD.dot(SVD.T).sum(axis=1)
    par_lens = np.array([len(toks) for toks in tokenized_pars])
    print('correlation of word freq with distance:', np.corrcoef(par_lens, dists)[0,1])
    print('norm of mean:', np.linalg.norm(SVD.mean(axis=0)))
    sortind = dists.argsort() # first is greatest dist
    dists[sortind][:5], dists[sortind[::-1]][:5]
    
    
    # get closest to center of field
    for i in range(2,0,-1):
        ind = sortind[i]
        d = np.linalg.norm(SVD[ind] - SVD.mean(axis=0))
        print('doc', ind, 'd =', d)
        print(par_texts[ind], end='\n\n')
        
    # get furthest from center of field
    for i in range(2):
        ind = sortind[::-1][i]
        d = np.linalg.norm(SVD[ind] - SVD.mean(axis=0))
        print('doc', ind, 'd =', d)
        print(par_texts[ind], end='\n\n')
    
'''
### Pointwise Mutual Information
The [pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information) calculates the level of association between two variables, document and word in this case (as designated by the 2-dimensional distribution), by controlling for both the frequency of a word and the number of words in a document. Higher values mean that the word is more uniquely associated with the document statistically than not.

Positive pointwise mutual information is a variant of PMI which sets negative values (words which are less associated with documents than expected) to zero. While we loose some information here, this solves the problem of -infinity values caused by taking log(0) and has shown to still be a robust measure.
Levy, Goldberg, Dagan (2015) _Improving Distributional Similarity with Lessons Learned from Word Embeddings_ ([link])(https://levyomer.files.wordpress.com/2015/03/improving-distributional-similarity-tacl-2015.pdf).
'''
    
'''
### Singular Value Decomposition of PPMI Matrix
This is simply decomposing the PPMI matrix into singular values, which we used to compress the matrix. Note that in cases where your vocab is larger than the number of documents, read the [implementation notes](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd).
'''
