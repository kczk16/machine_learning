from scipy.io import arff
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
from sklearn import metrics
import seaborn as sns


df = pd.read_csv('student-mat.csv', sep=';')
df.describe(include='all')
df = df.replace(['no', 'yes'], [0, 1])
df = df.rename({'school': 'is from MS'}, axis=1)
df = df.replace(['GP', 'MS'], [0, 1])
df = df.rename({'sex': 'is female'}, axis=1)
df = df.replace(['M', 'F'], [0, 1])
# age category: 1 - [15-17], 2 - [18-19], 3 - [20-22]
df['age'] = pd.cut(df['age'], bins=[14, 17, 19, 22], labels=[1, 2, 3])
df['age'] = pd.to_numeric(df['age'], downcast="integer")
df = df.rename({'address': 'is urban'}, axis=1)
df = df.replace(['R', 'U'], [0, 1])
df = df.rename({'famsize': 'is famsize greater than 3'}, axis=1)
df = df.replace(['LE3', 'GT3'], [0, 1])
df = df.rename({'Pstatus': 'parents together'}, axis=1)
df = df.replace(['A', 'T'], [0, 1])
# 1 - teacher, 2 - civil services, 3 - at home, 4 - other
df['Mjob'] = df['Mjob'].replace(['teacher', 'health', 'services', 'at_home', 'other'], [1, 2, 2, 3, 4])
df['Fjob'] = df['Fjob'].replace(['teacher', 'health', 'services', 'at_home', 'other'], [1, 2, 2, 3, 4])
# if the reason is related to schooling?
df = df.rename({'reason': 'reason related to schooling'}, axis=1)
df = df.replace(['home', 'other', 'reputation', 'course'], [0, 0, 1, 1])
df = df.replace(['mother', 'father', 'other'], [1, 2, 3])
df['G1'] = pd.cut(df['G1'], bins=[-1, 7, 14, 20], labels=[1, 2, 3])
df['G1'] = pd.to_numeric(df['G1'], downcast="integer")
df['G2'] = pd.cut(df['G2'], bins=[-1, 7, 14, 20], labels=[1, 2, 3])
df['G2'] = pd.to_numeric(df['G2'], downcast="integer")
df['G3'] = pd.cut(df['G3'], bins=[-1, 7, 14, 20], labels=[1, 2, 3])
df['G3'] = pd.to_numeric(df['G3'], downcast="integer")

corr = df.corr(method="pearson")
sns.heatmap(corr, xticklabels=True, yticklabels=True)

X = df.iloc[:, :-1]
Y = pd.DataFrame(df.iloc[:, -1])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=100)
clf = tree.DecisionTreeClassifier(random_state=0)
clf.fit(X_train, Y_train)
y_train_predicted = clf.predict(X_train)
y_test_predicted = clf.predict(X_test)
accuracy_score(Y_train, y_train_predicted)
accuracy_score(Y_test, y_test_predicted)
print('Train dataset accuracy: ', accuracy_score(Y_train, y_train_predicted))
print('Test dataset accuracy: ', accuracy_score(Y_train, y_train_predicted))

plt.figure(figsize=(16, 8))
tree.plot_tree(clf)
plt.show()

path = clf.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# fig, ax = plt.subplots()
# ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
# ax.set_xlabel("effective alpha")
# ax.set_ylabel("total impurity of leaves")
# ax.set_title("Total Impurity vs effective alpha for training set")


clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, Y_train)
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()


train_scores = [clf.score(X_train, Y_train) for clf in clfs]
test_scores = [clf.score(X_test, Y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(criterion='entropy', random_state=42)
rfc.fit(X_train, Y_train)
# Evaluating on Training set
rfc_pred_train = rfc.predict(X_train)
print('Training Set Evaluation F1-Score=>',accuracy_score(Y_train,rfc_pred_train))

# Evaluating on Test set
rfc_pred_test = rfc.predict(X_test)
print('Testing Set Evaluation F1-Score=>',accuracy_score(Y_test,rfc_pred_test))


from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10)
# estimator = DecisionTreeClassifier()
model = BaggingClassifier(n_estimators=100)
model.fit(X_train, Y_train)
model_pred_train = model.predict(X_train)
print('Training Set Evaluation F1-Score=>', accuracy_score(Y_train, model_pred_train))

# Evaluating on Test set
model_pred_test = model.predict(X_test)
print('Testing Set Evaluation F1-Score=>', accuracy_score(Y_test, model_pred_test))


from sklearn.ensemble import GradientBoostingClassifier

cv = KFold(n_splits=10)
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01)
model.fit(X_train, Y_train)
model_pred_train = model.predict(X_train)
print('Training Set Evaluation F1-Score=>', accuracy_score(Y_train, model_pred_train))

# Evaluating on Test set
model_pred_test = model.predict(X_test)
print('Testing Set Evaluation F1-Score=>', accuracy_score(Y_test, model_pred_test))

