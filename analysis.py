import matplotlib.pyplot as plt
import random
# SKLearn Libraries
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
# SKLearn Preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import label_binarize
# SKLearn Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# SKLearn Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.learning_curve import learning_curve
#These libraries are used to visualize the decision tree and require that you have GraphViz
#and pydot or pydotplus installed on your computer.
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.core.display import Image
import pydotplus as pydot
from patsy import dmatrices

def sample_rows(df, nrows):

	"""
	Function to return a sample of the total rows.
	
	Args:
		df (pd.DataFrame): DataFrame
		nrows (int): Number of rows to subset
	Returns:
		df (pd.DataFrame): Subsetted DataFrame
	"""

    rows = random.sample(df.index, nrows)
    df = df.ix[rows]
    return df

def create_dummies(df):

	"""
	Function to identify and create the dummies for the categorical variables.
	
	Args:
		df (pd.DataFrame): DataFrame
	Returns:
		df (pd.DataFrame): DataFrame with dummy columns
	"""

    y, X = dmatrices('''INTERNET ~ 
                     C(REGION) + 
                     C(ST) + 
                     NP + 
                     C(TYPE) + 
                     C(BATH) +
                     BDSP +
                     C(BLD) + 
                     C(BROADBND) +
                     C(FS) +
                     C(HANDHELD) +
                     C(LAPTOP) +
                     RMSP +
                     C(SINK) +
                     C(STOV) +
                     C(TEL) +
                     C(TEN) +
                     C(YBL) +
                     C(HHL) +
                     C(HHT) +
                     HINCP +
                     C(HUGCL) +
                     C(HUPAC) +
                     C(KIT) +
                     C(LNGI) +
                     C(MULTG) +
                     NOC +
                     C(NPP) +
                     C(NR) +
                     NRC +
                     C(PARTNER) +
                     C(PLM) +
                     C(PSF) +
                     C(R18) +
                     C(R60) +
                     C(R65) +
                     C(SSMC) +
                     C(SVAL) 
                     ''', df, return_type = 'dataframe')
    return y, X

def test_random_forest_n_estimators_parameter(params, X_train, X_test, y_train, y_test):

	"""
	Runs and plots a random forest estimator and allows for tuning of n_estimators.
	
	Args:
		params (list): list of parameters
	Returns:
		None
	"""

    x = []
    y = []
    z = []
    for param in params:
        x.append(param)
        model = RandomForestClassifier(n_estimators = param)
        fitted = model.fit(X_train, y_train)
        #Training Accuracy
        y.append(accuracy_score(y_train, fitted.predict(X_train)))
        #Test Accuracy
        z.append(accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(x, y)
    ts, = plt.plot(x, z)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc = 'best')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')

def test_random_forest_max_depth_parameter(params, X_train, X_test, y_train, y_test):

	"""
	Runs and plots a random forest estimator and allows for tuning of max_depth.
	
	Args:
		params (list): list of parameters
	Returns:
		None
	"""

    x = []
    y = []
    z = []
    for param in params:
        x.append(param)
        model = RandomForestClassifier(max_depth = param)
        fitted = model.fit(X_train, y_train)
        #Training Accuracy
        y.append(accuracy_score(y_train, fitted.predict(X_train)))
        #Test Accuracy
        z.append(accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(x, y)
    ts, = plt.plot(x, z)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc = 'best')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')

def test_random_forest_min_samples_split_parameter(params, X_train, X_test, y_train, y_test):

	"""
	Runs and plots a random forest estimator and allows for tuning of minimum samples split.
	
	Args:
		params (list): list of parameters
	Returns:
		None
	"""

    x = []
    y = []
    z = []
    for param in params:
        x.append(param)
        model = RandomForestClassifier(min_samples_split=param)
        fitted = model.fit(X_train, y_train)
        #Training Accuracy
        y.append(accuracy_score(y_train, fitted.predict(X_train)))
        #Test Accuracy
        z.append(accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(x, y)
    ts, = plt.plot(x, z)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc = 'best')
    plt.xlabel('min_samples_split')
    plt.ylabel('Accuracy')

def test_Gaussian_NB_estimators():

	"""
	Runs and plots a Gaussian Naive Bayes classifier.
	
	Args:
		None
	Returns:
		None
	"""

    model = GaussianNB()
    fitted = model.fit(X_train, y_train)
    #Training Accuracy
    y = (accuracy_score(y_train, fitted.predict(X_train)))
    #Test Accuracy
    z = (accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(range(1,20), [y] * 19)
    ts, = plt.plot(range(1,20), [z] * 19)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc = 'best')
    plt.xlabel('Gaussian_NB')
    plt.ylabel('Accuracy')

def test_Bernoulli_NB_estimators():

	"""
	Runs and plots a Bernoulli Naive Bayes classifier.
	
	Args:
		None
	Returns:
		None
	"""

    model = BernoulliNB()
    fitted = model.fit(X_train, y_train)
    #Training Accuracy
    y = (accuracy_score(y_train, fitted.predict(X_train)))
    #Test Accuracy
    z = (accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(range(1,20), [y] * 19)
    ts, = plt.plot(range(1,20), [z] * 19)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc = 'best')
    plt.xlabel('Bernoulli_NB')
    plt.ylabel('Accuracy')

def test_KNN_test_parameter(params, X_train, X_test, y_train, y_test):

	"""
	Runs and plots a K Nearest Neighbor classifier and allows for tuning of number neighbors.
	
	Args:
		params (list): list of parameters
	Returns:
		None
	"""

    x = []
    y = []
    z = []
    for param in params:
        x.append(param)
        model = KNeighborsClassifier(n_neighbors = param)
        fitted = model.fit(X_train, y_train)
        #Training Accuracy
        y.append(accuracy_score(y_train, fitted.predict(X_train)))
        #Test Accuracy
        z.append(accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(x, y)
    ts, = plt.plot(x, z)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc = 'best')
    plt.xlabel('# Neighbors')
    plt.ylabel('Accuracy')

def test_SVM_test_C_parameter(params, X_train, X_test, y_train, y_test):

	"""
	Runs and plots a support vector machine classifier and allows for tuning of C.
	
	Args:
		params (list): list of parameters
	Returns:
		None
	"""

    x = []
    y = []
    z = []
    for param in params:
        x.append(param)
        model = SVC(C = param, kernel = 'linear')
        fitted = model.fit(X_train, y_train)
        #Training Accuracy
        y.append(accuracy_score(y_train, fitted.predict(X_train)))
        #Test Accuracy
        z.append(accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(x, y)
    ts, = plt.plot(x, z)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc = 'best')
    plt.xlabel('C')
    plt.ylabel('Accuracy')

def test_SVM_rbf_test_C_parameter(params, X_train, X_test, y_train, y_test):

	"""
	Runs and plots a support vector machine classifier (rbf) and allows for tuning of C.
	
	Args:
		params (list): list of parameters
	Returns:
		None
	"""

    x = []
    y = []
    z = []
    for param in params:
        x.append(param)
        model = SVC(C = param, kernel = 'rbf')
        fitted = model.fit(X_train, y_train)
        #Training Accuracy
        y.append(accuracy_score(y_train, fitted.predict(X_train)))
        #Test Accuracy
        z.append(accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(x, y)
    ts, = plt.plot(x, z)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc = 'best')
    plt.xlabel('C')
    plt.ylabel('Accuracy')

def test_decision_tree_min_samples_split_parameter(params, X_train, X_test, y_train, y_test):

	"""
	Runs and plots a decision tree classifier and allows for tuning of minimum samples split.
	
	Args:
		params (list): list of parameters
	Returns:
		None
	"""

    x = []
    y = []
    z = []
    for param in params:
        x.append(param)
        model = DecisionTreeClassifier(min_samples_split = param)
        fitted = model.fit(X_train, y_train)
        #Training Accuracy
        y.append(accuracy_score(y_train, fitted.predict(X_train)))
        #Test Accuracy
        z.append(accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(x, y)
    ts, = plt.plot(x, z)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc = 'best')
    plt.xlabel('Min Samples Split')
    plt.ylabel('Accuracy')

def test_logistic_regression_c_parameter(params, X_train, X_test, y_train, y_test):

	"""
	Runs and plots a logistic regression classifier and allows for tuning of C.
	
	Args:
		params (list): list of parameters
	Returns:
		None
	"""

    x = []
    y = []
    z = []
    for param in params:
        x.append(param)
        model = LogisticRegression(C = param)
        fitted = model.fit(X_train, y_train)
        #Training Accuracy
        y.append(accuracy_score(y_train, fitted.predict(X_train)))
        #Test Accuracy
        z.append(accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(x, y)
    ts, = plt.plot(x, z)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc = 'best')
    plt.xlabel('C')
    plt.ylabel('Accuracy')

def plot_ROC_curve(model, X_train, X_test, y_train, y_test):

    """
    Function to plot an ROC curve
    """
    
    # Model Metrics
    print model
    print "*************************** Model Metrics *********************************"
    print 'Accuracy: %s' % cross_val_score(model, X_train, y_train, scoring = 'accuracy', cv = 5).mean()
    print 'Precision: %s' % cross_val_score(model, X_train, y_train, scoring = 'precision', cv = 5).mean()
    print 'Recall: %s' % cross_val_score(model, X_train, y_train, scoring = 'recall_weighted', cv = 5).mean()
    print 'F1: %s' % cross_val_score(model, X_train, y_train, scoring = 'f1', cv = 5).mean()

    fitted = model.fit(X_train, y_train)
    try:
        y_score = fitted.predict_proba(X_test)[:,1]
    except:
        y_score = fitted.decision_function(X_test)
  
    # Confusion matrix
    print "********************* Normalized Confusion Matrix *************************"
    cm = confusion_matrix(y_test, fitted.predict(X_test))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    
    # Classification Report
    print "********************* Classification Report********************************"    
    print classification_report(y_test, fitted.predict(X_test))
    
    print "********************* ROC Curve *******************************************"
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def plot_learning_curve(model, X_train, X_test, y_train, y_test):
    
    """
    Function to plot a learning curve
    """

    m, train_scores, valid_scores = learning_curve(estimator = model, 
                                                   X = X_train, y = y_train.ravel(), train_sizes = np.linspace(0.1,1.0, 80))

    train_cv_err = np.mean(train_scores, axis=1)
    test_cv_err = np.mean(valid_scores, axis=1)
    tr, = plt.plot(m, train_cv_err)
    ts, = plt.plot(m, test_cv_err)
    plt.legend((tr, ts), ('training error', 'test error'), loc = 'best')
    plt.title('Learning Curve')
    plt.xlabel('Data Points')
    plt.ylabel('Accuracy')