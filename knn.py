import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#read in the data using pandas
df = pd.read_csv('training.csv')
#df1=pd.DataFrame(df)
X = df
#X=x[x.columns[0:6]] 
y = df['J'].values

#split dataset into train and test data
#X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['G','H','I','J','K','L','M','N','O']), y, test_size=0.3, random_state=1,stratify=y)
X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1,stratify=y)
X_train = X_train1.drop(columns=['G','H','I','J','K','L','M','N','O'])
X_test = X_test1.drop(columns=['G','H','I','J','K','L','M','N','O'])
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 5)

# Fit the classifier to the data
knn.fit(X_train,y_train)
#a=knn.predict_proba(X_test)
a=knn.predict(X_test)
t=y_test
A=[]
B=[]
C=[]	
for i in range(len(y_test)):
  #print("X=%s, Predicted=%s" % (y_test[i], a[i]))
  if(a[i]==3):
    print("============= RAMAN ======================")
    print("X=%s, Y=%s , Z=%s" % (X_test1.iloc[i,6],X_test1.iloc[i,7],X_test1.iloc[i,8]))
    #print(X_test1.iloc[i,6])
    #print(X_test1['H'])
    #print(X_test1['I'])
    A.append(X_test1.iloc[i,6])
    B.append(X_test1.iloc[i,7])
    C.append(X_test1.iloc[i,8])
    print("============= SEHGAL ======================")
#df2 = pd.DataFrame({'X':A},{'Y':B},{'Z':C})
#df2.to_csv("test.csv")

df2 = pd.DataFrame(data={"X": A, "Y": B,"Z": C})
df2.to_csv("test.csv")
#print(a)
#check accuracy of our model on the test data
print("accuracy score",accuracy_score(a,t))

#np.savetxt("test.csv",a)

