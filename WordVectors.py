# Bag of Words Meets Bags of Popcorn problem
# Word Vectors approach (word2vec)
# Kalev Roomann-Kurrik
# Last Modified: June 29, 2015

# module imports
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk.data
from nltk.corpus import stopwords

# Read data from files
train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
test = pd.read_csv("testData.tsv", header = 0, delimiter = "\t", quoting = 3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)

# Verify the number of reviews that were read (100,000 in total)
print "Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size)

def review_to_wordlist(review, remove_stopwords=False):
  # Convert a document to a sequence of words, optionally removing stop words. Returns a list of words.

  # Remove HTML
  review_text = BeautifulSoup(review).get_text()

  # Remove non-alphanumeric characters
  review_text = re.sub("[^a-zA-Z0-9]", " ", review_text)

  # Convert words to lower case and split them
  words = review_text.lower().split()

  # Optionally, remove stop words (false by default)
  if remove_stopwords:
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]

  # Return a list of words
  return(words)

# Load the punkt tokenizer for splitting up sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer, remove_stopwords=False):
  # Split a review into parsed sentences. Returns a list of sentences, where each sentence is a list of words

  # Use the NLTK tokenizer to split the paragraph into sentences
  raw_sentences = tokenizer.tokenize(review.decode('utf-8').strip())

  # Loop over the sentences
  sentences = []

  for raw_sentence in raw_sentences:
    # If a sentence is empty, skip it
    if len(raw_sentence) > 0:
      # Else, call review_to_wordlist to get a list of words
      sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))

  # Return the list of sentences (each sentence is a list of words, so this returns a list of lists)
  return sentences

# Initialize an empty list of sentences
sentences = []

print "Parsing sentences from training set"
for review in train["review"]:
  sentences += review_to_sentences(review, tokenizer)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
  sentences += review_to_sentences(review, tokenizer)

print "Total number of sentences: %d\n" % (len(sentences)) 
