# Import needed libraries
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from numpy import mean, std
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score, \
    cross_validate, StratifiedKFold, LeaveOneOut
from sklearn import metrics, svm
from tabulate import tabulate

# import the data
samplesInfo = pd.read_csv('samplesInfo.csv')
brainData = pd.read_csv('QuantRegBrainData.csv')
geneData = pd.read_csv('gene.f.with.entrez.csv')

# when you use brainData.csv, put this as a comment
brainData = brainData.drop(columns=['Unnamed: 0'])

# when you use other csvs, put this as a comment
# brainData = brainData.drop(columns=['Unnamed: 0', 'Name'])

# Sort both data frames according to the sample ID
samplesInfo = samplesInfo.sort_values(by='SAMPID')
brainData = brainData.reindex(sorted(brainData.columns), axis=1)

brainData = brainData.T
labels = np.array(samplesInfo['DTHHRDY'])

################################# Feature selection with SelectPercentile ##############################################
# If you don"t want to run this feature selection method, skip this part and continue to splitting the data

features = SelectPercentile(f_classif, percentile=0.5)
brainData = features.fit_transform(brainData, samplesInfo['DTHHRDY'])
feNames = features.get_feature_names_out()

#get the genes number
for i in range(len(feNames)):
    feNames[i] = feNames[i][1:]
feNames = [int(i, base=16) for i in feNames]


# set parameters
paramsLGBM = {
    'max_depth':range(2,10,2),
    'min_child_weight':range(1,10,2),
    'scale_pos_weight': [0.5, 1, 2, 5, 10, 20],
    'feature_fraction': [0.5, 0.7, 0.8, 0.9],
    'max_bin': range(3,10,2),
    #'num_leaves': range(4, 150, 2),
}
########################################################################################################################

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(brainData,
                                                                            samplesInfo['DTHHRDY'],
                                                                            test_size=0.2,
                                                                            random_state=42)
# Undersampling
# Randomly under sample the majority class
rus = RandomUnderSampler(random_state=1)
X_train, y_train = rus.fit_resample(train_features, train_labels)


# ////////////////////////////// Feature selection with logistic regression ////////////////////////////////////////////
# If you don"t want to run this feature selection method, skip this part and continue to running models

log_reg_lasso = LogisticRegression(random_state=1)
parameters = {'penalty':['l1'], # L1 penalty = Lasso
              'C': np.exp(np.arange(-5,5,0.1)), # small C values means stronger regularization
              'solver': ['liblinear', 'saga'],  # For L1 penalty one must choose between 'liblinear' or 'saga' solver.
              'max_iter': [1000]} # Increasing the number of iterations to improve convergence

res = GridSearchCV(log_reg_lasso, parameters, cv=5, n_jobs=-1, verbose=3).fit(X_train, y_train) # X is scaled

lasso_coef = res.best_estimator_.fit(X_train, y_train).coef_[0]
Coef_dict = {Variable: [round(Coef,3)] for Variable, Coef in zip(brainData.columns, lasso_coef)}
Non_zero = {Variable: [round(Coef,3)] for Variable, Coef in zip(brainData.columns, lasso_coef) if Coef != 0}
Zero = [Variable for Variable, Coef in zip(brainData.columns, lasso_coef) if Coef == 0]

Non_zero5 = pd.DataFrame(Non_zero, index = ['Lasso Coefficient']).T
Zero = pd.DataFrame(Zero, columns = ['Coefficients that were shurnked to zero'])

brainData = brainData.iloc[:, np.array(Non_zero5.index.values),]

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(brainData,
                                                                            samplesInfo['DTHHRDY'],
                                                                            test_size=0.2,
                                                                            random_state=42)
# Undersampling
# Randomly under sample the majority class
rus = RandomUnderSampler(random_state=1)
X_train, y_train = rus.fit_resample(train_features, train_labels)
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# ************************************ LGBM ***********************************

###### Without randomizedSearch: ######
lgbm = LGBMClassifier(seed=42)
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)

# Accuracy for the test data
test_pred= lgbm.predict(test_features)
print("\nPrediction on the testing data")
print("The confusion matrix:\n",confusion_matrix(test_labels, test_pred))
print("Accuracy of the test:", metrics.accuracy_score(np.array(test_labels), np.round(test_pred)))
print("Sensitivity of the test:", metrics.recall_score(np.array(test_labels), np.round(test_pred), pos_label=2))
print("Specificity of the test:", metrics.recall_score(np.array(test_labels), np.round(test_pred), pos_label=1))

# Accuracy for the train data
train_pred= lgbm.predict(X_train)
print("\nPrediction on the training data")
print("Accuracy train:", metrics.accuracy_score(np.array(y_train), np.round(train_pred)))
print("The confusion matrix:\n",confusion_matrix(y_train, train_pred))

# #### With randomizedSearch #####

##########DO NOT RUN FOR SELECT PERCENTILE#############
# set parameters
paramsLGBM = {
    'max_depth':range(2,10,2),
    'min_child_weight':range(1,10,2),
    'scale_pos_weight': [0.5, 1, 2, 5, 10, 20],
    'feature_fraction': [0.5, 0.7, 0.8, 0.9],
    'max_bin': range(3,10,2),
}
#######################################################

lgbm_classifier = LGBMClassifier(seed=42)
random_search_lgbm = RandomizedSearchCV(lgbm_classifier, param_distributions=paramsLGBM, n_iter=500, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3, random_state = 42)
random_search_lgbm.fit(X_train, y_train)

test_predLGBM = random_search_lgbm.predict(test_features)
print("\nPrediction on the testing data")
print("The confusion matrix:\n",confusion_matrix(test_labels, test_predLGBM))
print("Accuracy of test:", metrics.accuracy_score(np.array(test_labels), np.round(test_predLGBM)))
print("Sensitivity of the test:", metrics.recall_score(np.array(test_labels), np.round(test_predLGBM), pos_label=2))
print("Specificity of the test:", metrics.recall_score(np.array(test_labels), np.round(test_predLGBM), pos_label=1))

# Accuracy for the train data
train_pred= random_search_lgbm.predict(X_train)
print("\nPrediction on the training data")
print("Accuracy:", metrics.accuracy_score(np.array(y_train), np.round(train_pred)))
print("The confusion matrix:\n",confusion_matrix(y_train, train_pred))

# !!!!!!!!!!!! Feature selection by using the feature importance !!!!!!!!!!!!!!!!!!!!!!!!!!!!

top_params_lgbm = random_search_lgbm.best_params_
#run the LGBMClassifier with the best params found fron the randomized search
lgbm_model = LGBMClassifier(scale_pos_weight=top_params_lgbm['scale_pos_weight'], min_child_weight=top_params_lgbm["min_child_weight"],
                           max_depth=top_params_lgbm["max_depth"], max_bin=top_params_lgbm["max_bin"], feature_fraction=top_params_lgbm["feature_fraction"])
lgbm_model.fit(X_train, y_train)

lgbmImp = lgbm_model.feature_importances_
impDF = pd.DataFrame(lgbmImp)
filtDF = impDF[impDF.iloc[:, 0] > 0]

#Check intersection
np.intersect1d(np.array(Non_zero.index.values), np.array(filtDF.index.values))

#create newBrainData only with the important
newBrainData = brainData.iloc[:,np.array(filtDF.index.values)]

#Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(newBrainData,
                                                                            samplesInfo['DTHHRDY'],
                                                                            test_size=0.2,
                                                                            random_state=42)
# Undersampling
# Randomly under sample the majority class
rus = RandomUnderSampler(random_state=1)
X_train, y_train = rus.fit_resample(train_features, train_labels)

# Fit the model wuth the important features
lgbm_classifier = LGBMClassifier(seed=42)
random_search_lgbm = RandomizedSearchCV(lgbm_classifier, param_distributions=paramsLGBM, n_iter=100, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3, random_state = 42)


random_search_lgbm.fit(X_train, y_train)

test_predLGBM = random_search_lgbm.predict(test_features)
print("\nPrediction on the testing data")
print("The confusion matrix:\n", confusion_matrix(test_labels, test_predLGBM))
print("Accuracy of test:", metrics.accuracy_score(np.array(test_labels), np.round(test_predLGBM)))
print("Sensitivity of the test:", metrics.recall_score(np.array(test_labels), np.round(test_predLGBM), pos_label=2))
print("Specificity of the test:", metrics.recall_score(np.array(test_labels), np.round(test_predLGBM), pos_label=1))

# Accuracy for the train data
train_pred= random_search_lgbm.predict(X_train)
print("\nPrediction on the training data")
print("Accuracy:", metrics.accuracy_score(np.array(y_train), np.round(train_pred)))
print("The confusion matrix:\n",confusion_matrix(y_train, train_pred))

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ************************************ SVM ***********************************

svm_model = svm.SVC(kernel='linear', C = 10)
svm_model.fit(X_train, y_train)
# Accuracy for the test data
test_pred_svm= svm_model.predict(test_features)
print("\nPrediction on the testing data")
print("The confusion matrix:\n", confusion_matrix(test_labels, test_pred_svm))
print("Accuracy test:", metrics.accuracy_score(np.array(test_labels), np.round(test_pred_svm)))
print("Sensitivity of the test:", metrics.recall_score(np.array(test_labels), np.round(test_pred_svm), pos_label=2))
print("Specificity of the test:", metrics.recall_score(np.array(test_labels), np.round(test_pred_svm), pos_label=1))


# Accuracy for the train data
train_pred_svm= svm_model.predict(X_train)
print("\nPrediction on the training data")
print("Accuracy train:", metrics.accuracy_score(np.array(y_train), np.round(train_pred_svm)))
print("The confusion matrix:\n",confusion_matrix(y_train, train_pred_svm))

param_grid_svm = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                   ]

svm_mod = svm.SVC()
grid_search_svm = RandomizedSearchCV(svm_mod, param_distributions=param_grid_svm, n_jobs=-1, cv=5, random_state = 1)
grid_search_svm.fit(X_train, y_train)
# Accuracy for the test data
test_pred_svm= grid_search_svm.predict(test_features)
print("\nPrediction on the testing data")
print("The confusion matrix:\n", confusion_matrix(test_labels, test_pred_svm))
print("Accuracy test:", metrics.accuracy_score(np.array(test_labels), np.round(test_pred_svm)))
print("Sensitivity of the test:", metrics.recall_score(np.array(test_labels), np.round(test_pred_svm), pos_label=2))
print("Specificity of the test:", metrics.recall_score(np.array(test_labels), np.round(test_pred_svm), pos_label=1))


# Accuracy for the train data
train_pred_svm= svm_model.predict(X_train)
print("\nPrediction on the training data")
print("Accuracy train:", metrics.accuracy_score(np.array(y_train), np.round(train_pred_svm)))
print("The confusion matrix:\n",confusion_matrix(y_train, train_pred_svm))

######### CROSS - VALIDATION ###########

# Undersampling
rus = RandomUnderSampler(random_state=1)
X, y = rus.fit_resample(brainData, samplesInfo['DTHHRDY'])


# !!!! LGBM !!!!!!

top_params_lgbm = random_search_lgbm.best_params_
#run the LGBMClassifier with the best params found from the randomized search

### for 13,000 + logistic featere selection
lgbm_model = LGBMClassifier(scale_pos_weight=5, min_child_weight=2,
                           max_depth=8, max_bin=7, random_state=1)
##

### for Precentile feature selection
lgbm_model = LGBMClassifier(scale_pos_weight=top_params_lgbm['scale_pos_weight'], min_child_weight=top_params_lgbm["min_child_weight"],
                           max_depth=top_params_lgbm["max_depth"], max_bin=top_params_lgbm["max_bin"], feature_fraction=top_params_lgbm["feature_fraction"])

##

#with underSampling - KFold
max_score = [0, 0, 0]
for k in range(5, 21):
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    score = cross_val_score(lgbm_model, X, y, cv=kf, n_jobs=-1)
    if max_score[1] < mean(score):
        max_score = [k, mean(score), std(score)]

print('K :%.3f Accuracy: %.3f (%.3f)' % (max_score[0], max_score[1], max_score[2]))

#without undersampling - StratifiedKFold
max_score = [0, 0, 0]
for k in range(5, 21):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    score = cross_val_score(lgbm_model, brainData, labels, cv=kf, n_jobs=-1)
    if max_score[1] < mean(score):
        max_score = [k, mean(score), std(score)]

print('K :%.3f Accuracy: %.3f (%.3f)' % (max_score[0], max_score[1], max_score[2]))

# !!!! SVM !!!!!!

svm_model = svm.SVC(kernel='linear', C=25)

max_score = [0,0,0]
#with underSampling - KFold
for k in range(5, 21):
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    score = cross_val_score(svm_model, X, y, cv=kf, n_jobs=-1)
    if max_score[1] < mean(score):
        max_score = [k, mean(score), std(score)]

print('K :%.3f Accuracy: %.3f (%.3f)' % (max_score[0], max_score[1], max_score[2]))

#without undersampling - StratifiedKFold
max_score = [0,0,0]
for k in range(5, 21):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    score = cross_val_score(svm_model, brainData, labels, cv=kf, n_jobs=-1)
    if max_score[1] < mean(score):
        max_score = [k, mean(score), std(score)]

print('K :%.3f Accuracy: %.3f (%.3f)' % (max_score[0], max_score[1], max_score[2]))


# GET FEATURE IMPORTANCE - SVM

# for all samples + linear svm
svm_model = svm.SVC(kernel='linear', C=25, random_state=1)
svm_model.fit(X_train, y_train)

# Get the feature importance
svm_imp = svm_model.coef_[0]

#sort the importance and take top 500 (or less if we dont have more that 500)
svmImpDF = pd.DataFrame(svm_imp)
svmImpDF = svmImpDF.sort_values(by=0)
highestTenAllImp = svmImpDF[0].nlargest(n=500)

#get the information of the important genes
brainData = pd.read_csv('QuantRegBrainData.csv').T
highestTenAllGenes =brainData.iloc[:, highestTenAllImp.index.values]
highestTenAll = pd.DataFrame(highestTenAllGenes.iloc[0,:])
highestTenAll.rename(columns = {'Unnamed: 0':'ID'}, inplace = True)
highestTenAll['Importance'] = highestTenAllImp

# get the name og the important genes
topGenes = geneData.loc[geneData['Name'].isin(np.array(highestTenAll['ID'])), :]
topGenes = topGenes.sort_values(by='Name')
highestTenAll = highestTenAll.sort_values(by='ID')
highestTenAll['Name'] = np.array(topGenes['Description'])
highestTenAll = highestTenAll.sort_values(by='Importance', ascending=False)

print(tabulate(highestTenAll, headers='keys', tablefmt='psql', showindex=False))

#print the gene names to put it in enricher
list = np.array(highestTenAll['Name'])
for elem in list:
        print (elem)