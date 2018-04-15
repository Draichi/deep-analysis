import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survical (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblbings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel('titanic.xls')
# print(df.head())
original_df = pd.DataFrame.copy(df)
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
# print(df.head())

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
# print(df.head())

# increase accuracy
df.drop(['ticket', 'sex'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

# create a empty column on original_df
original_df['cluster_group'] = np.nan

# we gonna iterate through the labels and populate
# the values in cluster_group column as those labels
for i in range(len(X)):
    # the i row of original_df under the column cluster group
    # we gonna set that value = to labels[i]
    # iloc reference the row in our dataframe
    original_df['cluster_group'].iloc[i] = labels[i]

# how many unique labels do we have
n_clusters_ = len(np.unique(labels))

survival_rates = {}

for i in range(n_clusters_):
    # we're creating a temporary datafreme which only
    # where cluster_group is cluster_group[i]
    temp_df = original_df[(original_df['cluster_group']==float(i))]
    # survival df would be where tmp_df who survived
    survival_cluster = temp_df[(temp_df['survived']==1)]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
# 0, 1, 2... are the groups
print(original_df[original_df['cluster_group']==0].describe())
