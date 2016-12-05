from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import cross_validation
import numpy as np
import scipy as sp
import pandas as pd

def log_loss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def main():
    #read in data, parse into training and target sets
    dataset = pd.read_csv('Data/train.csv')
    # print(dataset.head(5))
    # print(dataset.describe())

    target = dataset["Activity"]
    train = dataset.ix[:,1:]

    alg = RandomForestClassifier(n_estimators=100, n_jobs=4)
    # alg = GradientBoostingClassifier(random_state=1, n_estimators=50, verbose=1, max_depth=3, min_samples_split=8, min_samples_leaf=4)

    #Simple K-Fold cross validation. 5 folds.
    cv = cross_validation.KFold(len(train), n_folds=5, shuffle=False)

    #iterate through the training and test cross validation segments and
    #run the classifier on each one, aggregating the results into a list
    results = []
    for traincv, testcv in cv:
        predict_proba = alg.fit(train.iloc[traincv], target.iloc[traincv]).predict_proba(train.iloc[testcv])
        results.append(log_loss(target.iloc[testcv], [x[1] for x in predict_proba]) )

    #print out the mean of the cross-validated results
    print ("Results: " + str( np.array(results).mean() ))

    # create submission on test data
    test = pd.read_csv('Data/test.csv')
    alg.fit(train, target)
    predict_test = [x[1] for x in alg.predict_proba(test)]
    
    # print ("Number of predictions: ")
    # print (len(predict_test))
    size = len(predict_test)
    index = list(x+1 for x in range(size))
    #HEADER:  MoleculeId, PredictedProbability
    submission = pd.DataFrame({
        "MoleculeId":index,
        "PredictedProbability": predict_test
    })
    submission.to_csv("Submission/kaggle.csv", index=False)
    
if __name__=="__main__":
    main()