from sklearn.datasets import fetch_20newsgroups
import sys
import os
import glob
import pandas as pd

def get_newsgroup_data(useN=100):
    
    nd = fetch_20newsgroups(shuffle=True, random_state=0)
    texts, fnames =  nd['data'][:useN], nd['filenames'][:useN]
    fnames = [fn.split('/')[-1] for fn in fnames]
    
    return texts, fnames


def make_textfiles(useN=100, target_dir = 'tmp'):
    
    # make dir if it doesn't exist
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
        print('Created new dir', target_dir)
    
    # delete .txt files if they already exist
    files = glob.glob(target_dir + '/' + '*.txt')
    i = 0
    for fname in files:
        os.remove(fname)
        i += 1
    print('Removed', i, 'files that previously existed.')
    
    nd = fetch_20newsgroups(shuffle=True, random_state=0)
    texts, fnames =  nd['data'][:useN], nd['filenames'][:useN]

    for fn,t in zip(fnames,texts):
        with open(target_dir + '/'+fn.split('/')[-1]+'.txt', 'w') as f:
            f.write(t)
            
    print('Added', len(texts), 'new files to', target_dir)
    

def make_spreadsheet(useN=100, target_fname = 'tmp_example.csv'):
    
    nd = fetch_20newsgroups(shuffle=True, random_state=0)
    texts, fnames =  nd['data'][:useN], nd['filenames'][:useN]
    fnames = [fn.split('/')[-1] for fn in fnames]
    
    df = pd.DataFrame(index=range(len(fnames)), columns=['text',])
    for i, fn,t in zip(range(len(fnames)),fnames,texts):
        df.loc[i,'title'] = fn
        df.loc[i,'text'] = t
     
    df.to_csv(target_fname, index=False)
    
    print('saved texts to', target_fname)
    
    
    

if __name__ == '__main__':
    
    if not len(sys.argv) > 1:
        raise Exception('Need to provde command from {spreadsheet, textfiles, singletext}')
    
    cmd_type = sys.argv[1]
    if len(sys.argv) > 2:
        useN = int(sys.argv[2])
    else:
        useN = 100
        
    # run commands
    if cmd_type == 'spreadsheet':
        make_spreadsheet(useN,)
        
    elif cmd_type == 'textfiles':
        make_textfiles(useN,)
    
    else:
        raise Exception('Enter a valid command! Options: {spreadsheet, textfiles}')