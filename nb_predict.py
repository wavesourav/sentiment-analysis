import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer

############# Importing trained classifier and fitted vectorizer ################
nb_clf = pickle.load(open("nb_clf_crude_oil", 'rb'))
vectorizer = pickle.load(open("vectorizer_crude_oil", 'rb'))

############## Predict sentiment using the trained classifier ###################

# Import test data set
data_pred = pd.read_csv("CrudeOil_News_Articles_test.csv", encoding="ISO-8859-1")
X_test = data_pred.iloc[:, 1]  # extract column with news articles
X_vec_test = vectorizer.transform(X_test)  # use transform() instead of fit_transform()
X_tfidf_test = TfidfTransformer().fit_transform(X_vec_test)

# Convert to dense matrix
X_tfidf_test = X_tfidf_test.toarray()

# Predict the sentiment values
y_pred = nb_clf.predict(X_tfidf_test)
