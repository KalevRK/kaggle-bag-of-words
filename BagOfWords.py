# Bag of Words Meets Bags of Popcorn problem
# Kalev Roomann-Kurrik
# Last Modified: June 17, 2015

# module imports
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
# nltk.download() # Download text data sets, including stop words
from nltk.corpus import stopwords
# print stopwords.words("english")

# read in the labeled training data
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# print train.shape
# print train.columns.values
# print train["review"][0]

# initialize the BeautifulSoup object on a single movie review
example1 = BeautifulSoup(train["review"][0])

# print the raw review and then the output of get_text(), for comparison
# print train["review"][0]
# print example1.get_text()

# use regex to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]", " ", example1.get_text())
# print letters_only
lower_case = letters_only.lower()
words = lower_case.split()

# remove stop words from "words"
words = [w for w in words if not w in stopwords.words("english")]
print words
