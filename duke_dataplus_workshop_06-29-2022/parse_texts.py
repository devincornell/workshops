import pandas as pd
import spacy
import pickle

if __name__ == '__main__':
    df = pd.read_csv('data/exit_survey.csv', skiprows=[1,2], usecols=['Q73', 'Q74', 'Q75', 'Q76.1'])
    df = df.fillna('')
    #df = df.iloc[:3]
    good_things = list(df['Q73'])
    bad_things = list(df['Q74'])
    slowed_things = list(df['Q75'])
    rec_things = list(df['Q76.1'])

    nlp = spacy.load('en_core_web_trf') # load a language model

    full_dataset = {
        'good': [nlp(response) for response in good_things],
        'bad': [nlp(response) for response in bad_things],
        'slowed': [nlp(response) for response in slowed_things],
        'rec': [nlp(response) for response in rec_things],
    }

    with open('data/parsed_text.pic', 'wb') as f:
        pickle.dump(full_dataset, f)

