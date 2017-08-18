from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import svm
from sklearn.metrics import accuracy_score

def preprocess(X,y):
    ### test_size is the percentage of events assigned to the test set
    ### (remainder go into training)
    features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2, random_state=42)


   ### text vectorization--go from strings to lists of numbers
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)
    joblib.dump(vectorizer, 'vectorizer.pkl')

   ### feature selection, because text is super high dimensional and
    ### can be really computationally chewy as a result
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    joblib.dump(selector, 'selector.pkl')
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()
    return features_train_transformed, features_test_transformed, labels_train, labels_test

with open("../dataset/data_final_inflado_intencion.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]


def cleanPhrase(phrase):
    return phrase.replace("!","").replace(".","").replace(",","").replace(":","").replace("?","").lower()

features = [cleanPhrase(frase.split("\t")[1]) for frase in content]
targets = [frase.split("\t")[0] for frase in content]

features_train, features_test, labels_train, labels_test=preprocess(features,targets)

lin_svc = svm.LinearSVC(C=100).fit(features_train, labels_train)

print(accuracy_score(labels_test, lin_svc.predict(features_test)))



from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=30).fit(features_train, labels_train)
print(accuracy_score(labels_test, rfc.predict(features_test)))

new_phrase="you are feeling so happy"
vect=joblib.load("vectorizer.pkl")
select=joblib.load('selector.pkl')
features_transformed  = vect.transform([new_phrase])
features = select.transform(features_transformed).toarray()
# print(lin_svc.predict(features))
