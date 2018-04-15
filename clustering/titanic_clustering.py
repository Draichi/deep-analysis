import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

'''
[1] => Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
[2] => survival Survical (0 = No; 1 = Yes)
[3] => name Name
[4] => sex Sex
[5] => age Age
[6] => sibsp Number of Siblbings/Spouses Aboard
[7] => parch Number of Parents/Children Aboard
[8] => ticket Ticket Number
[9] => fare Passenger Fare (British pound)
[10] => cabin Cabin
[11] => embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
[12] => boat Lifeboat
[13] => body Body Identification Number
[14] => home.dest Home/Destination
'''

xl = pd.ExcelFile('titanic.xls')
df = xl.parse('titanic3')
df.drop(['body', 'name'], 1, inplace=True)
df.fillna(0, inplace=True)
print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values
    # we're going to each column, defining convert_to_int function
    # then, we're asking if that column is a number, if not,
    # we convert to list, get the set of that list, we gona take
    # the unique elemnts, populate that dictionare, and then
    # we're setting the value of df[column] by mapping
    # convert_to_int function to the value that's in that column
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            # this will return the index of that val.
            # ex.: 'Female' index 0 will return:
            # text_digit_vals = {'Female': 0}
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            # this will gone get all unique and non repetitive values there
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int, df[column]))
    return df
df = handle_non_numerical_data(df)
print(df.head())

# descomment the line above to increase accuracy
# df.drop(['ticket', 'sex'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    # we never use this in supervised learning
    # because here, it just gonna make this prediction based
    # on the location of that centroid
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

# the order of cluster is totally arbitrary, so if we had
# 0.28 is the same of 0.72
# the order is just inverted
# print(df.head())
print('Accuracy:',correct/len(X))