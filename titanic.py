import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv("titanic.csv")
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

df[list(pd.get_dummies(df["Embarked"]).columns)] = pd.get_dummies(df["Embarked"])
df.fillna({"Embarked": "S"}, inplace=True)
df.drop("Embarked", axis=1, inplace=True)

age_1 = df[df["Pclass"] == 1]["Age"].median()
age_2 = df[df["Pclass"] == 2]["Age"].median()
age_3 = df[df["Pclass"] == 3]["Age"].median()

print(age_1, age_2, age_3)

def fill_age(row):
    if pd.isnull(row["Age"]):
        if row["Pclass"] == 1:
            return age_1
        if row["Pclass"] == 2:
            return age_2
        return age_3
    return row["Age"]

df["Age"] = df.apply(fill_age, axis=1)

def fill_Gender(Gender):
    if Gender == "male":
        return 1
    return 0

df["Gender"] = df["Gender"].apply(fill_Gender)

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=60)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print('Відсоток правильно передбачених результатів: ', accuracy_score(y_test, y_pred) * 100)
print('Confusion matrix: ')
print(confusion_matrix(y_test, y_pred))


