# Bag of Words Meets Bags of Popcorn problem
# Kalev Roomann-Kurrik
# Last Modified: June 27, 2015

# module imports
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
# nltk.download() # Download text data sets, including stop words
from nltk.corpus import stopwords
# print stopwords.words("english")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# read in the labeled training data
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# clean and pre-process a single movie review
# input: single string (a raw movie review)
# output: single string (a preprocessed movie review)
def review_to_words(raw_review):
  # Remove HTML
  review_text = BeautifulSoup(raw_review).get_text()

  # Remove non-letters
  letters_only = re.sub("[^a-zA-Z]", " ", review_text)

  # Convert to lower case and tokenize into individual words
  words = letters_only.lower().split()

  # Convert stop words list into a set (faster to search in Python)
  stops = set(stopwords.words("english"))

  # Remove stop words
  meaningful_words = [w for w in words if not w in stops]

  # Join the words back into one string separated by space, and return the result
  return(" ".join(meaningful_words))

# Get number of reviews based on the dataframe column size
num_reviews = train["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

print "Cleaning and parsing the training set of moview reviews...\n"

# Loop over each review
for i in xrange(0, num_reviews):
  if((i+1) % 1000 == 0):
    print "Review %d of %d\n" % (i+1, num_reviews)
  # Call our function for each review and add the result to the list of clean reviews
  clean_train_reviews.append(review_to_words(train["review"][i]))

print "Creating the bag of words...\n"

# Initialize the "CountVectorizer" object
vectorizer = CountVectorizer(analyzer = "word", \
                             tokenizer = None, \
                             preprocessor = None, \
                             stop_words = None, \
                             max_features = 5000)

# Use the fit_transform() method to fit the model and learn the vocabulary, and also transform the training dat into feature vectors. The input to fit_transform should be a list of strings
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an array
train_data_features = train_data_features.toarray()

print "Training the random forest..."

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as features and the sentiment labels as the response variable
forest = forest.fit(train_data_features, train["sentiment"])

# Read the test data
test = pd.read_csv("testData.tsv", header = 0, delimiter = "\t", quoting = 3)

# Verify that there are 25,000 rows and 2 columns
print test.shape

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = []

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0, num_reviews):
  if ((i+1) % 1000 == 0):
    print "Review %d of %d\n" % (i+1, num_reviews)
  clean_review = review_to_words(test["review"][i])
  clean_test_reviews.append(clean_review)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})

# Use pandas to write the comma-separated output file
output.to_csv("Bag_of_Words_model.csv", index = False, quoting = 3)
