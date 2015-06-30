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
import logging
from gensim.models import word2vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Configure logging for word2vec output
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)

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

# Set values for word2vec parameters
num_features = 300 # Word vector dimensionality
min_word_count = 40 # Minimum word count
num_workers = 4 # Number of threads to run in parallel
context = 10 # Context window size
downsampling = 1e-3 # Downsample setting for frequent words

# Initialize and train the word2vec model
print "Training model..."
model = word2vec.Word2Vec(sentences, workers = num_workers, size = num_features, min_count = min_word_count, window = context, sample = downsampling)

# If you don't plan to train the model any further, calling init_sims will make the model much more memory-efficient
model.init_sims(replace = True)

# Can save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)

def makeFeatureVec(words, model, num_features):
  # Average all of the word vectors in a given paragraph

  # Pre-initialize an empty numpy array (for speed)
  featureVec = np.zeros((num_features,), dtype = "float32")
  nwords = 0

  # Convert index2word (a list that contains the names of the words in the model's vocabulary) to a set for speed
  index2word_set = set(model.index2word)

  # Loop over each word in the review, and if it's in the model's vocabulary, add its feature vector to the total
  for word in words:
    if word in index2word_set:
      nwords = nwords + 1
      featureVec = np.add(featureVec, model[word])

  # Divide the result by the number of words to get the average
  featureVec = np.divide(featureVec, nwords)
  return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
  # Given a set of reviews (each one a list of words), calculate the average feature vector for each one and return a 2D numpy array

  # Initialize a counter
  counter = 0

  # Preallocate a 2D numpy array, for speed
  reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype = "float32")

  # Loop through the reviews
  for review in reviews:
    # Print a status message every 1000th review
    if counter % 1000. == 0.:
      print "Review %d of %d" % (counter, len(reviews))

    # Call the function that makes average feature vectors
    reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

    # Increment the counter
    counter = counter + 1

  return reviewFeatureVecs

# Calculate the average feature vectors for training and testing sets, using the functions defined above. Remove stop words to cut down on noise.

clean_train_reviews = []

for review in train["review"]:
  clean_train_reviews.append(review_to_wordlist(review, remove_stopwords = True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

print "Creating average feature vecs for test reviews"

clean_test_reviews = []

for review in test["review"]:
  clean_test_reviews.append(review_to_wordlist(review, remove_stopwords = True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

# Fit a random forest to the training data, using 100 trees
forest = RandomForestClassifier(n_estimators = 100)

print "Fitting a random forest to labeled training data..."

forest = forest.fit(trainDataVecs, train["sentiment"])

# Test & extract results
result = forest.predict(testDataVecs)

# Write the test results
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("Word2Vec_AverageVectors.csv", index = False, quoting = 3)
