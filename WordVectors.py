# Bag of Words Meets Bags of Popcorn problem
# Word Vectors approach (word2vec)
# Kalev Roomann-Kurrik
# Last Modified: June 29, 2015

# module imports
import pandas as pd

# Read data from files
train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
test = pd.read_csv("testData.tsv", header = 0, delimiter = "\t", quoting = 3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)

# Verify the number of reviews that were read (100,000 in total)
print "Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size)

