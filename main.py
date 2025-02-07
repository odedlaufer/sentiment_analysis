# -*- coding: utf-8 -*-
"""
Matan Abucasis
Oded Laufer
"""

# importing packages for pre-processing & sentiment analyzing 

import pandas as pd
import re
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from string import punctuation

# importing packages for evaluation of model & plotting the results
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt



df = pd.read_csv("fwc22_tweets.csv")
df2 = df.drop(df.columns[[0,1,2,3,]], axis=1)

tweets_only = df2.to_numpy()[:,0]
analysis = df2.to_numpy()[:,1]

#to work with
tweets_only_5 = tweets_only[:27]

# regex pattern to remove hashtags, mentions, urls
regex = "#\w*|@\w*|http\S+"

# lemmatizer object
lemmatizer = WordNetLemmatizer()

# stop words for tokenizing
stop_words = stopwords.words('english')

# removing mentions, hashtags & urls
def regex_tweet(tweets):

    for tweet in range(len(tweets)):
        tweets[tweet] = ''.join(re.sub(regex,"", tweets[tweet]))

# Removing punctutation
def remove_punc(tweets):

    for tweet in range(len(tweets)):
        tweets[tweet] = ''.join(tw for tw in tweets[tweet] if tw not in punctuation)


# Lemmatization & POS Tagging process #
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None


#calculate the negative, positive, neutral and compound scores, plus verbal evaluation
def sentiment_vader(sentence):

    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = "Positive"

    elif sentiment_dict['compound'] <= - 0.05 :
        overall_sentiment = "Negative"

    else :
        overall_sentiment = "Neutral"
  
    return negative, neutral, positive, compound, overall_sentiment


### Process ###

regex_tweet(tweets_only_5)
remove_punc(tweets_only_5)

tokens_arr = []
analysis_arr = []


# Tokenizing
for i in range(len(tweets_only_5)):

    tokens = word_tokenize(tweets_only_5[i])
    new_tokens = [t.lower() for t in tokens if t.lower() not in stop_words and len(t) > 1]
    tokens_arr.insert(i, new_tokens)


 # Semantic Analyize for each tokens
idx = 0
for tokens in tokens_arr:

    # Set initial tags => (word, tag)
    pos_tagged = pos_tag(tokens)

    # Fixing the tags (N, V, ADJ...)
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

    # Lemmatize the sentence
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:       
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

    lemmatized_sentence = " ".join(lemmatized_sentence)
    
    # Semantic Analyze output
    sentiment_senctence = sentiment_vader(lemmatized_sentence)
    
    # Append the sentence, predicted sentiment & actual sentiment
    analysis_arr.append((lemmatized_sentence, sentiment_senctence, analysis[idx]))
    
    idx = idx + 1

# printing the array
# for i in range(len(analysis_arr)):
#     print(analysis_arr[i][0], " ||", analysis_arr[i][1][4], " ||", analysis[i].capitalize())


# converting back to pandas DataFrame
df_output = pd.DataFrame(analysis_arr, columns=['Sentence', 'Prediction', 'Actual'])
# print(df_output)


### Evaluation ###

labels = ['positive', 'neutral', 'negative']
y_true = analysis
y_pred = [i[1][4].lower() for i in analysis_arr]

# compute Confusion Matrix
confusin_mat = confusion_matrix(y_true, y_pred, labels=labels)

dispMat = ConfusionMatrixDisplay(confusion_matrix=confusin_mat, display_labels=labels)
# dispMat.plot()
# plt.show()

eval_scores = classification_report(y_true, y_pred, digits=3)
print(f"Precision, Recall, F1 Scores: \n{eval_scores}")

accuracy = accuracy_score(y_true, y_pred)
print(f"The accuracy is: {accuracy * 100}%")
















 

