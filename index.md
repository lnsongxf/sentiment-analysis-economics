# Sentiment Analysis of Economic Reports Using Logistic Regression

Sentiment analysis is a hot topic in NLP, but this technology is increasingly relevant in the financial markets - which is in large part driven by investor sentiment.

With so many reports and economic bulletins being generated on a daily basis, one of the big challenges for policymakers is to extract meaningful information in a short period of time to inform policy decisions.

In this example, two reports from the European Central Bank website (available from the relevant GitHub repository) are converted into text format, and then a logistic regression is used to rank keywords by positive and negative sentiment. The bulletins in question are sourced from the [European Central Bank website](https://www.ecb.europa.eu/pub/economic-bulletin/html/index.en.html).

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

## CountVectorizer

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

Next, the CountVectorizer is used to both learn the text provided and then transform this text into a sparse matrix for analysis.

```
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(train2)
X = cv.transform(train2)
X_test = cv.transform(test2)
```

As an example, X is now defined as follows:

```
<35000x3111 sparse matrix of type '<class 'numpy.int64'>'
	with 33996 stored elements in Compressed Sparse Row format>
```
  
# Logistic Regression

Having properly formatted the text using the procedures above, the logistic regression is now trained using the train set - with 75% of this train set being used to train the model, and the remaining 25% reserved for validation purposes.

```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [1 if i < 5000 else 0 for i in range(35000)]

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("C regularization parameter = %s yields accuracy of %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))
```

Here are the accuracy results given each setting for the regularization parameter C:

```
C regularization parameter = 0.01 yields accuracy of 0.8587428571428571
C regularization parameter = 0.05 yields accuracy of 0.8587428571428571
C regularization parameter = 0.25 yields accuracy of 0.8587428571428571
C regularization parameter = 0.5 yields accuracy of 0.8578285714285714
C regularization parameter = 1 yields accuracy of 0.8603428571428572
```

Now, let's assess the overall model accuracy:

```
>>> final_model = LogisticRegression(C=1)
>>> final_model.fit(X, target)
>>> print ("Model Accuracy: %s" 
>>>        % accuracy_score(target, final_model.predict(X_test)))

Model Accuracy: 0.8568857142857143
```

## Sentiment Analysis

Now that the logistic regression has been trained, let's observe the 20 top positive and negative words as indicated by the model.

```
feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}
for highest_positives in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:20]:
    print (highest_positives)
    
for highest_negatives in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:20]:
    print (highest_negatives)
```

### Positive Words

```
('emes', 2.8087942720630537)
('uncertainties', 2.6969534994438056)
('council', 2.6406758309641223)
('governing', 2.6406758309641223)
('weigh', 2.59457201762457)
('medium', 2.572699507530095)
('moderate', 2.5668344198303923)
('spreads', 2.4637286879924543)
('remain', 2.419420988993332)
('currencies', 2.4157254834188198)
('relations', 2.415447222800664)
('activity', 2.3397460113102984)
('amid', 2.3326302368928875)
('environment', 2.3326302368928875)
('near', 2.263318676133264)
('pace', 2.236958259190495)
('vis', 2.236958259190495)
('stimulus', 2.195135454801049)
('projected', 2.1116174471453855)
('expected', 2.047692902832495)
```

### Negative Words

```
('between', -1.7022023009702862)
('capital', -1.6995634610512402)
('not', -1.693279073829643)
('flows', -1.60129753927606)
('ils', -1.5333183320103416)
('implied', -1.522046232039325)
('model', -1.4989591800406523)
('factor', -1.487131947235996)
('cycle', -1.4395412126145506)
('stock', -1.4304293384442213)
('only', -1.3847531339874612)
('circulation', -1.356685868110847)
('such', -1.342224894450962)
('one', -1.2969805371162153)
('option', -1.2812312352865205)
('banknotes', -1.2486347387958983)
('international', -1.2144512592278498)
('returns', -1.2144512592278498)
('no', -1.1785121839560957)
('cash', -1.1719971887652905)
```

We can see that certain words have been classed as positive and negative - which is highly dependent on context since a word can be interpreted as both positive or negative on this basis.

As an example, **currencies** has been interpreted as a positive word, implying that the currency markets may have shown positive performance (from an ECB perspective) for the month in question. On the other hand, **stock** is ranked as having negative sentiment, indicating that stock market performance may have been lagging over the same period.

## Conclusion

In this example, it has been demonstrated how a text can be summarized and sentiment analysis generated through a logistic regression. While NLP methods are not inherently foolproof and long texts still require a degree of human interpretation to ensure proper understanding, sentiment analysis can be of significant use when it comes to text summarization and allowing a reader to digest the main points of a report quickly.

You can find the relevant GitHub Repository here, and I also recommend [the following post](https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184) if you wish to gain a further understanding of sentiment analysis.
