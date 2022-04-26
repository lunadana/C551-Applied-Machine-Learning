import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

from NaiveBayes import NaiveBayes, evaluate_acc

import warnings
warnings.filterwarnings("ignore")   # suppresses future deprecation warnings - too annoying


#######################################################################
###################### K FOLD CROSS VALIDATION ########################
#######################################################################


def cross_validation_split(df_x, df_y, k):
    df_x = pd.DataFrame(df_x)
    df_y = pd.DataFrame(df_y)
    df_x_new = df_x.copy()
    df_y_new = df_y.copy()
    n = df_x.shape[0]
    sub_size = int(n/k)
    TBR_x = [df_x_new] * k
    TBR_y = [df_y_new] * k
    for i in range(k-1):
        TBR_x[i] = df_x_new.iloc[:sub_size,:]
        TBR_y[i] = df_y_new.iloc[:sub_size,:]
        df_x_new = df_x_new.iloc[sub_size:,:]
        df_y_new = df_y_new.iloc[sub_size:,:]
    TBR_x[k-1] = df_x_new
    TBR_y[k-1] = df_y_new
    return TBR_x, TBR_y


def kfoldCV(x, y, model):
    k = len(x)
    avg_acc = 0
    for i in range(k):
        validation_set_x = x[i]
        validation_set_y = y[i]
        train_set_x = pd.DataFrame()
        train_set_y = pd.DataFrame()
        for j in range(k):
            if j != i:
                train_set_x = train_set_x.append(x[j])
                train_set_y = train_set_y.append(y[j])

        print(f'Running {model.__class__.__name__} using set {i} as validation set.')
        model.fit(train_set_x.to_numpy(), train_set_y.to_numpy().flatten())
        y_prob = model.predict(validation_set_x.to_numpy())

        try:
            y_pred = np.argmax(y_prob, axis=1)  # selects the label with the highest likelihood for each instance in NB
        except np.AxisError:
            y_pred = y_prob  # logistic regression

        accuracy = evaluate_acc(validation_set_y.to_numpy().flatten(), y_pred)
        avg_acc += accuracy
        print(f"Accuracy is {accuracy}\n")
        print("#########################################\n")

    return avg_acc / k

######################################################################
############################## SCRIPT ################################
######################################################################

if __name__ == '__main__':
    """
    selects the dataset to run the model on:
    1 -> 20Newsgroups dataset
    2 -> Sentiment140 dataset
    """
    dataset = 2

    # /!\ UNCOMMENT THE MODEL YOU WANT TO USE /!\
    #model = NaiveBayes
    model = LogisticRegression

    # opens & stores the stop words as a list
    with open("stopwords.txt") as f:
        stop_words = f.read()
        stop_words = stop_words.split('\n')

    if dataset == 1:  # 20 news groups dataset
        twenty_train = fetch_20newsgroups(subset='train',
                                          shuffle=True,
                                          remove=(['headers', 'footers', 'quotes']))

        vectorizer = CountVectorizer(max_features=5000, stop_words=stop_words)
        X_counts = vectorizer.fit_transform(twenty_train.data)  # creates count matrix

        tfidf_transformer = TfidfTransformer()
        X_tfidf = tfidf_transformer.fit_transform(X_counts)

        # test-train split
        x_train, x_test, y_train, y_test = model_selection.train_test_split(X_tfidf, twenty_train.target,
                                                                            test_size=0.2)

        x_train, x_test = x_train.toarray(), x_test.toarray()

    else:  # sentiment140 dataset
        Sentiment140_test = pd.read_csv('data/testdata.manual.2009.06.14.csv', encoding='ISO-8859-1',
                                        header=None)
        Sentiment140_test = Sentiment140_test.loc[Sentiment140_test[0] != 2]  # removes instances with label = 2
        num_of_test_instances = Sentiment140_test.shape[0]
        Sentiment140 = pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1',
                                   header=None).sample(10000)

        # appends the train & test datasets together to vectorize them identically & will be split after
        Sentiment140 = Sentiment140.append(Sentiment140_test)

        Sentiment_columns = ['Y', 'id', 'date', 'query', 'user', 'text']
        Sentiment140.columns = Sentiment_columns

        # replaces all labels of 4 with 1, to respect the model's implementation requiring labels to be consecutive
        Sentiment140['Y'].replace({4: 1}, inplace=True)

        vectorizer = CountVectorizer(max_features=2000, stop_words=stop_words)
        X_counts = vectorizer.fit_transform(Sentiment140[['text']].values.flatten().tolist())

        tfidf_transformer = TfidfTransformer()
        X_tfidf = tfidf_transformer.fit_transform(X_counts).toarray()

        # splits the combined data back into test and train sets as they were given
        x_train = X_tfidf[:-num_of_test_instances, :]
        x_test = X_tfidf[-num_of_test_instances:, :]

        y_train = Sentiment140[['Y']].values[:-num_of_test_instances, :].flatten()
        y_test = Sentiment140[['Y']].values[-num_of_test_instances:, :].flatten()

    # KFOLD CV
    k = 5
    x, y = cross_validation_split(x_train, y_train, k)
    kfoldCV_acc = kfoldCV(x, y, model())
    print(f'Average accuracy of the {model.__name__} model on {k}-fold CV is {kfoldCV_acc}.')

    model_instance = model()
    model_instance.fit(x_train, y_train)
    y_prob = model_instance.predict(x_test)

    try:
        y_pred = np.argmax(y_prob, axis=1)  # selects the label with the highest likelihood for each instance in NB
    except np.AxisError:
        y_pred = y_prob  # logistic regression

    accuracy = evaluate_acc(y_test, y_pred)
    print(f'Average accuracy of the {model.__name__} model on the whole train set and test set is {accuracy}.')
