# Sentiment Analysis Project

This project is a sentiment analysis implementation using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. It analyzes the sentiment of textual data, such as tweets, and classifies it as positive, neutral, or negative. The project involves pre-processing the text, performing sentiment analysis, and evaluating the model's performance.

**Introduction**

Sentiment analysis is the process of determining the sentiment expressed in a piece of text, which can be positive, negative, or neutral. This sentiment analysis project utilizes the VADER sentiment analysis tool along with various natural language processing techniques to classify the sentiment of tweets. The project focuses on pre-processing the text, performing sentiment analysis, and evaluating the model's performance.

**Installation**

To run this project, you need to have the following dependencies installed:

pandas
nltk
vaderSentiment
scikit-learn
matplotlib
You can install these dependencies using the following command:

```
pip install pandas nltk vaderSentiment scikit-learn matplotlib
```

**Usage**

Clone the repository or download the source code.
Place your dataset file (e.g., fwc22_tweets.csv) in the same directory as the Python script.
Open the Python script file (sentiment_analysis.py).
Modify the script if necessary (e.g., adjust file paths, dataset columns).
Run the script

**Results**

The sentiment analysis project processes the provided dataset, fwc22_tweets.csv, which contains tweets and their corresponding sentiment labels. The project performs the following steps:

**Pre-processing:**

Removes hashtags, mentions, and URLs from the tweets.
Removes punctuation from the tweets.
Tokenization:
Tokenizes the pre-processed tweets.
Converts the tokens to lowercase.
Removes stop words.
Lemmatization:
Performs part-of-speech tagging on the tokens.
Lemmatizes the tokens based on their respective part-of-speech tags.

**Sentiment Analysis:**

Utilizes the VADER sentiment analysis tool to calculate the negative, neutral, positive, and compound scores for each lemmatized sentence.
Assigns an overall sentiment label to each sentence (positive, neutral, or negative).

**Evaluation:**
Computes the confusion matrix to evaluate the model's performance.
Displays the confusion matrix and plots it using matplotlib.
Prints the precision, recall, and F1 scores for each sentiment class.
Prints the accuracy of the sentiment analysis model.
Evaluation
The sentiment analysis model's performance is evaluated using the following metrics:

Confusion Matrix: The confusion matrix shows the number of true positive, true negative, false positive, and false negative predictions for each sentiment class.
Precision: Precision is the ratio of correctly predicted instances of a sentiment class to the total predicted instances of that class.
Recall: Recall is the ratio of correctly predicted instances of a sentiment class to the total instances of that class in the dataset.
F1 Score: The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both measures.
Accuracy: Accuracy is the overall performance of the model, calculated as the ratio of correctly predicted instances to the total instances in the dataset.
The evaluation results are displayed in the console.

# Pre-Proccessing code snippet

```
# Process the dataset
df = pd.read_csv("fwc22_tweets.csv")
df2 = df.drop(df.columns[[0,1,2,3,]], axis=1)

tweets_only = df2.to_numpy()[:,0]
analysis = df2.to_numpy()[:,1]

# Pre-processing functions
regex = "#\w*|@\w*|http\S+"
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

# Remove mentions, hashtags, and URLs
def regex_tweet(tweets):
    for tweet in range(len(tweets)):
        tweets[tweet] = ''.join(re.sub(regex, "", tweets[tweet]))

# Remove punctuation
def remove_punc(tweets):
    for tweet in range(len(tweets)):
        tweets[tweet] = ''.join(tw for tw in tweets[tweet] if tw not in punctuation)
```

**Contributors**

This sentiment analysis project was developed by:

Matan Abucasis

Oded Laufer
