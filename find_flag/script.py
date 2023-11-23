import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn import tree

#https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
df= pd.read_csv("flag.data", names = cols)

#variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange', 'mainhue', 'bars','stripes', 'circles','crosses', 'saltires','quarters','sunstars','triangle','animate']

cont_1 = 'N.America'
cont_2 = 'S.America'
cont_3 = 'Europe'
cont_4 = 'Africa'
cont_5 = 'Asia'
cont_6 = 'Oceania'

continent_dict = {1:cont_1, 2:cont_2, 3:cont_3, 4:cont_4, 5:cont_5, 6:cont_6}
#Print number of countries by landmass, or continent
print(df.groupby('landmass')['name'].count())

#Create a new dataframe with only flags from Europe and Oceania
europe_oceania = df[df['landmass'].isin([3, 6])]

#Print the average values of the predictors for Europe and Oceania
# print(europe_oceania.groupby('landmass')[var].mean())

#Create labels for only Europe and Oceania
labels = df['landmass']

#Print the variable types for the predictors
print(df[var].dtypes)

#Create dummy variables for categorical predictors
data = pd.get_dummies(df[var])

#Split data into a train and test set
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state = 1)

#Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21)
acc_depth = []

dtree = DTC(random_state=1)
for depth in depths:
    dtree.set_params(max_depth = depth)
    dtree.fit(x_train, y_train)
    acc_depth.append(dtree.score(x_test, y_test))


#Plot the accuracy vs depth using seaborn
sns.set_style("whitegrid")
plt.plot(depths, acc_depth)
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.show()

#Find the largest accuracy and the depth this occurs
max_acc = max(acc_depth)

#Refit decision tree model with the highest accuracy and plot the decision tree
dtree.set_params(max_depth = acc_depth.index(max_acc)+1)
dtree.fit(x_train, y_train)
plt.figure(figsize=(15,10))  # Increase the size of the figure
tree.plot_tree(dtree, fontsize=10)  # Increase the font size
plt.show()

#Create a new list for the accuracy values of a pruned decision tree.  Loop through
#the values of ccp and append the scores to the list
ccp = dtree.cost_complexity_pruning_path(x_train, y_train)
ccp_alpha = ccp['ccp_alphas']
acc_ccp = []
for ccpa in ccp_alpha:
    dtree.set_params(ccp_alpha = ccpa)
    dtree.fit(x_train, y_train)
    acc_ccp.append(dtree.score(x_test, y_test))


#Plot the accuracy vs ccp_alpha
sns.set_style("whitegrid")
plt.plot(ccp_alpha, acc_ccp)
plt.xlabel("ccp_alpha")
plt.ylabel("Accuracy")
plt.show()

# Find the largest accuracy and the ccp value this occurs
max_acc = max(acc_ccp)
ccp_max = ccp_alpha[acc_ccp.index(max_acc)]

# Find the depth that gave the maximum accuracy
max_depth = acc_depth.index(max(acc_depth)) + 1

# Fit a decision tree model with the values for max_depth and ccp_alpha found above
dtree.set_params(max_depth=max_depth, ccp_alpha=ccp_max)
dtree.fit(x_train, y_train)

# Plot the final decision tree with larger font size
plt.figure(figsize=(15,10))  # Increase the size of the figure
tree.plot_tree(dtree, fontsize=10)  # Increase the font size
plt.show()

new_flag_attributes = {
    'red': [1],  # Red is present
    'green': [0],  # Green is not present
    'blue': [1],  # Blue is present
    'gold': [0],  # Gold is not present
    'white': [1],  # White is present
    'black': [0],  # Black is not present
    'orange': [0],  # Orange is not present
    'mainhue': ['red'],  # Main color can be considered blue
    'bars': [0],  # No bars
    'stripes': [3],  # 13 stripes
    'circles': [0],  # No circles
    'crosses': [0],  # No crosses
    'saltires': [0],  # No saltires
    'quarters': [0],  # No quarters
    'sunstars': [0],  # 50 stars
    'triangle': [0],  # No triangles
    'animate': [0]  # No animate features
}

# Convert to DataFrame
new_flag_df = pd.DataFrame(new_flag_attributes)

# Convert categorical variables to dummy variables
new_flag_df = pd.get_dummies(new_flag_df)

# Make sure the new data has the same features as the training data, fill missing with 0s
new_flag_df = new_flag_df.reindex(columns = x_train.columns, fill_value=0)

# Predict the landmass
predicted_landmass = dtree.predict(new_flag_df)

# Print the predicted landmass using the dictionary created earlier
print('Predicted landmass: ', continent_dict[predicted_landmass[0]])