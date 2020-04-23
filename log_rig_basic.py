# https://www.kaggle.com/rtatman/datasets-for-regression-analysis

#The Ultimate Halloween Candy Power Ranking:
# Can you predict if a candy is chocolate or not based on its other features?

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


#data


candy_data = pd.read_csv(r'/home/ram/Downloads/kaggle/candy-data.csv')

print(candy_data.info())


# Lets see 0, 1 numbers of chocolate as bar
labels=("candy_has_no_chocolate","candy_has_chocolate")
candy_data['chocolate'].value_counts().plot.barh();
plt.yticks([0, 1], labels)
plt.xlabel('chocolate')


plt.show()



# "competitorname" feature we dont need and lets drop it
candy_data.drop("competitorname", inplace = True, axis=1)

y = candy_data.chocolate.values
X = candy_data.drop(["chocolate"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X_train, y_train)
accuracy = loj.score(X_test, y_test)
print("Accuracy on test data for a given model is {}".format(accuracy))
# predict
y_pred = loj_model.predict(X_test)
# confussion matrix
print("confussion matrix/n")
print(confusion_matrix(y_test, y_pred))

# classification report

print("classification report/n")

print(classification_report(y_test, y_pred))

# addding boosting
#from sklearn.ensemble import AdaBoostRegressor

###n_acc=abr.score(X_test,y_test)
#print("boosting:Accuracy on test data for a given model is {}".format(n_acc))

#predictions=abr.predict(X_test)
#d=[]
#for i in range(len(predictions)):
#    if predictions[i]>0.5:
#        d.append(1)
#    else:
#        d.append(0)
#print(d)
#n_acc=abr.score(X_test,y_test)
#print("boosting:Accuracy on test data for a given model is {}".format(n_acc))
#print(classification_report(y_test,d))

#
#Visualization
sns.set(style="whitegrid")
g = sns.regplot(x="winpercent", y="chocolate", data=candy_data, fit_reg=False)
g.figure.set_size_inches(10, 10)
plt.show()

sns.set(style="whitegrid")
g = sns.regplot(x="winpercent", y="chocolate", data=candy_data, fit_reg=True)
g.figure.set_size_inches(10, 10)
plt.show()