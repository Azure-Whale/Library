import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
# Read Data
rate_table = pd.read_excel('Data/Item_based.xlsx',index_col=0)
# Basic Table Info
rate_table = rate_table.T
X = rate_table.iloc[0:4,0:3]
X=X.astype('int')
Y = rate_table.iloc[0:4,3]
Y=Y.astype('int')
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, Y)
test_X=rate_table.iloc[4,0:3]
testX = test_X.astype('int')
test_X = test_X.values.reshape(1,-1)
print(test_X)
print(neigh.predict(test_X))