# Sentiment Analysis of Economic Reports Using Logistic Regression

Sentiment analysis is a hot topic in NLP, but this technology is increasingly relevant in the financial markets - which are in large part driven by investor sentiment.

With so many reports and economic bulletins being generated on a daily basis, one of the big challenges for policymakers is to extract meaningful information in a short period of time to inform policy decisions.

In this example, two reports from the European Central Bank website (available from the relevant GitHub repository) are converted into text format, and then a logistic regression is used to rank keywords by positive and negative sentiment.

## Converting PDF to text

Firstly, pdf2txt is used to convert the pdf files into text format using a Linux shell.

```
pdf2txt.py -o eb201806.txt eb201806.en.pdf
pdf2txt.py -o eb201807.txt eb201807.en.pdf
```

## Word Tokenization with NLTK

The two texts are inputted - the first report as the **training set** and the second as the **test set**, i.e. the eventual logistic regression is "trained" on the first report, and then predictions are made on the second.

```
train = open('eb201806.txt').read()
test = open('eb201807.txt').read()

import re
train2=re.sub('[^A-Za-z]+', ' ', train)
test2=re.sub('[^A-Za-z]+', ' ', test)
```

The next step is **word tokenization**, which involves splitting a large sample of text into words.

```
# WORD TOKENIZATION: Splitting a large sample of text into words
from nltk.corpus import stopwords

# Set of stopwords needs to be downloaded when running on the first occasion
import nltk
# nltk.download('stopwords')

from nltk.tokenize import word_tokenize
print(word_tokenize(train2))
train2=word_tokenize(train2)
train2

print(word_tokenize(test2))
test2=word_tokenize(test2)
test2
```

Here is a sample:

```
['Economic', 'Bulletin', 'Issue', 'Contents', 'Economic', 'and', 'monetary', 'developments', 'Overview', 'External', 'environment' .... 'Trends', 'and', 'developments', 'in', 'the', 'use', 'of', 'euro', 'cash', 'over', 'the', 'past', 'ten', 'years', 'Statistics', 'ECB']
```

## CountVectorizer and Logistic Regression

The train and test set have the following lengths:

```
>>> len(train2)
43900

>>> len(test2)
39417
```

Each is partitioned so as to have an equal 35,000 length:

```
train2=train2[:35000]
test2=test2[:35000]
```

