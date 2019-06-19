
# Intro to Text Analysis

This tutorial was created as a workshop for the [Broom Center for Demography](http://www.broomcenter.ucsb.edu/). The tutorial is meant to give an overview of text analysis using Spacy and NLTK. While there are many more features to these packages than I showed here, I hope they will be helpful for introducing you to the kinds of things people do with text analysis tools.

## Preliminary

This project works with Python 3 and the Spacy and NLTK packages.

## Get Started

The first step in running the workshop is to run the script dump_data.py. This command will pull data from python packages and place them as text into either a .csv or text files (in the provided folder). The command takes a few user arguments and looks like this:

`python dump_data.py <corpus> <outfilename> [-n <number of docs>]`

**corpus**: I have prepared three corpora for use in this workshop: 'brown','gutenberg', or 'newsgroup'.

**outfilename**: This should be either a .csv file or a folder.

**number of docs**: Put the number of docs you would like to run your test code on. The smaller the number, the smaller your dataset will be. I recommend starting with a very low number for getting started. Some of the analyses may take alot of time with big datasets.

## Run Notebooks

Follow the notebooks 0-2 to see the examples I provided. course_dmos.ipynb is the notebook I used live in the workshop, and empath_example.ipynb was used to demonstrate the python Empath library upon request from one of the workshop participants.

