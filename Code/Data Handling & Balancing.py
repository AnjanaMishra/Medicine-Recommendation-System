
import numpy as np
import pandas as pd
import re
import string
import spacy
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd

df = pd.read_pickle("450k_rows_cleaned.pkl")

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])



stemmer = SnowballStemmer('english')

def review_to_words(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', raw_review)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords
    meaningful_words = [w for w in words if not w in stops]
    # 6. Stemming
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(stemming_words))

# create a list of stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
parser = English()
punctuations = string.punctuation
# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens



df['review_clean'] = df['description'].apply(review_to_words)


import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,2))
#  tf-idf vector
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)




# part 1---vader sentiment analyzer for c_review
analyzer = SentimentIntensityAnalyzer()
# create new col vaderReviewScore based on C-review
req_df['vaderReviewScore'] = req_df['review_clean'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# define the positive, neutral and negative
positive_num = len(req_df[req_df['vaderReviewScore'] >=0.05])
neutral_num = len(req_df[(req_df['vaderReviewScore'] >-0.05) & (req_df['vaderReviewScore']<0.05)])
negative_num = len(req_df[req_df['vaderReviewScore']<=-0.05])

# create new col vaderSentiment based on vaderReviewScore
req_df['vaderSentiment'] = req_df['vaderReviewScore'].map(lambda x:int(2) if x>=0.05 else int(1) if x<=-0.05 else int(0) )
req_df['vaderSentiment'].value_counts() # 2-pos: 99519; 1-neg: 104434; 0-neu: 11110

# label pos/neg/neu based on vaderSentiment result
req_df.loc[req_df['vaderReviewScore'] >=0.05,"vaderSentimentLabel"] ="positive"
req_df.loc[(req_df['vaderReviewScore'] >-0.05) & (req_df['vaderReviewScore']<0.05),"vaderSentimentLabel"]= "neutral"
req_df.loc[req_df['vaderReviewScore']<=-0.05,"vaderSentimentLabel"] = "negative"



req_df.head()


req_df['vaderReviewScore'].max()


req_df['vaderReviewScore'].min()



#Normalizing "vaderReviewScore"
criteria = [req_df['vaderReviewScore'].between(-0.9977, -0.620), req_df['vaderReviewScore'].between(-0.619, -0.260), req_df['vaderReviewScore'].between(-0.259, 0.103), req_df['vaderReviewScore'].between(0.104, 0.460), req_df['vaderReviewScore'].between(0.461, 0.9969)]
values = [1, 2, 3,4,5]

req_df['normalVaderScore'] = np.select(criteria, values, 0)


req_df['meanNormalizedScore'] = (req_df['Overall rating'] + req_df['normalVaderScore'])/2
req_df.head()



#Sorting Data based on Condition and then by drug
grouped = req_df.groupby(['Drug Name.1']).agg({'meanNormalizedScore' : ['mean']})
grouped.to_csv('Grouped_Drug_Recommendation_Normalized')
grouped.head(20)



from sklearn.model_selection import train_test_split

x_train, x_test = train_test_split(req_df, test_size = 0.25, random_state = 0)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

cv = CountVectorizer(max_features = 20000, ngram_range = (4, 4))
pipeline = Pipeline([('vect',cv)])

x_train_features = pipeline.fit_transform(x_train['review_clean'])
x_test_features = pipeline.fit_transform(x_test['review_clean'])

print("x_train_features :", x_train_features.shape)
print("x_test_features :", x_test_features.shape)
# let's make a new column review sentiment

req_df.loc[(req_df['meanNormalizedScore'] >= 3), 'Review_Sentiment'] = 1
req_df.loc[(req_df['meanNormalizedScore'] < 3), 'Review_Sentiment'] = 0

req_df['Review_Sentiment'].value_counts()




ros = RandomOverSampler()
train_x, train_y = ros.fit_resample(np.array(req_df['review_clean']).reshape(-1, 1), np.array(req_df['Review_Sentiment']).reshape(-1, 1));
train_os = pd.DataFrame(list(zip([x[0] for x in train_x], train_y)), columns = ['review_clean', 'Review_Sentiment']);



len(train_os['review_clean'])




from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, confusion_matrix, recall_score
from sklearn.metrics import classification_report
# making our dependent variable



features = train_os['review_clean'] # the features we want to analyze
labels = train_os['Review_Sentiment'] # the labels, we want to test against
X_train, X_test, y_train, y_test= train_test_split(features, labels, test_size=0.2, random_state=0)
clf2 = LogisticRegression(random_state=0,solver='lbfgs',max_iter=2000,multi_class='auto')
#clf3 = SVC(kernel="linear", C=5)
clf7 = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=10, tol=None)