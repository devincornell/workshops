import sys
import os
import glob
import pandas as pd
import argparse

import data_sources

def make_textfiles(texts, textnames, target_dir):
    
    # make dir if it doesn't exist
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    
    # delete .txt files if they already exist
    files = glob.glob(target_dir + '/' + '*.txt')
    i = 0
    for fname in files:
        os.remove(fname)
        i += 1
    print('Removed', i, 'files that previously existed.')

    for fn,t in zip(textnames,texts):
        with open(target_dir + '/'+fn.split('/')[-1]+'.txt', 'w') as f:
            f.write(t)
    

def make_spreadsheet(texts, textnames, target_fname):
    
    
    df = pd.DataFrame(index=range(len(textnames)), columns=['text','name'])
    for i, t,n in zip(range(len(textnames)), texts, textnames):
        df.loc[i,'name'] = n
        df.loc[i,'text'] = t
    
    df.to_csv(target_fname, index=False)
    
    
    
    
def get_parser():
    parser = argparse.ArgumentParser(description='Dump data for example text processing.')
    parser.add_argument('corpus', choices=('brown','gutenberg','newsgroup'), help='Format of output.')
    #parser.add_argument('datatype', choices=('spreadsheet','textfiles'), help='Format of output.')
    parser.add_argument('outfile', type=str, help='Folder or csv file for output.')
    parser.add_argument('-n','--numdocs', type=int, required=False, default=100, help='Number of docs to output.')
    return parser
    

if __name__ == '__main__':
    
    # build command line interface
    parser = get_parser()
    args = parser.parse_args()

        
    # retrieve data
    if args.corpus == 'brown':
        texts, names = data_sources.get_brown_data(args.numdocs)
    
    elif args.corpus == 'gutenberg':
        texts, names = data_sources.get_gutenberg_data(args.numdocs)
    
    elif args.corpus == 'newsgroup':
        texts, names = data_sources.get_newsgroup_data(args.numdocs)
        
    # save output data
    if args.outfile.endswith('.csv'):
        make_spreadsheet(texts,names,args.outfile)
        print('saved texts to', args.outfile)
        
    else:
        make_textfiles(texts,names,args.outfile)
        print('saving output into', args.outfile)
        