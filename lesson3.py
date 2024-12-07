import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from datetime import datetime

df = pd.read_csv("train.csv")
print(df.info())

gender_group = df.groupby("gender")["result"].mean()
print(gender_group)

relation_group = df.groupby("relation")["result"].mean()
print(relation_group)

df["bdate"] = pd.to_datetime(df["bdate"], errors="coerce")
df["bdate_year"] = df["bdate"].dt.year.fillna(df["bdate"].dt.year.median())

current_year = 2024
df["age"] = current_year - df["bdate_year"]

df["is_employed"] = df["career_end"] > df["career_start"]

df['last_seen'] = pd.to_datetime(df['last_seen'], dayfirst=True, errors='coerce')
current_date = datetime.now()
df['days_since_last_seen'] = (current_date - df['last_seen']).dt.days

df.drop(["id", "last_seen", "langs", "bdate", "career_end", "career_start"], axis=1, inplace=True)

def remove_False_life_main(row):
    if row["life_main"] == "False":
        return -1
    return int(row["life_main"])
def remove_False_people_main(row):
    if row["people_main"] == "False":
        return -1
    return int(row["people_main"])

df["life_main"] = df.apply(remove_False_life_main, axis=1)
df["people_main"] = df.apply(remove_False_people_main, axis=1)

df = pd.get_dummies(df, columns=["education_form", "education_status", "occupation_type"], drop_first=True)

print(df.info())

X = df.drop("result", axis=1)
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print("Відсоток правильно передбачених результатів:",
      accuracy_score(y_test, y_pred) * 100)
print("Confusion matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(cm[0][0], "- правильно класифіковані як ті, хто не придбав курс")
print(cm[0][1], "- помилково класифіковані як ті, хто придбав курс, хоча насправді вони його не придбали")
print(cm[1][0], "- помилково класифіковані як ті, хто не придбав курс, хоча насправді вони його придбали")
print(cm[1][1], "- правильно класифіковані як ті, хто придбав курс")

df2 = pd.read_csv("test.csv", encoding="utf-8", delimiter=";")

df2['last_seen'] = pd.to_datetime(df2['last_seen'], dayfirst=True, errors='coerce')
current_date = datetime.now()
df2['days_since_last_seen'] = (current_date - df2['last_seen']).dt.days

ID = df2['id']

df2["bdate"] = pd.to_datetime(df2["bdate"], errors="coerce")
df2["bdate_year"] = (df2["bdate"].
                     dt.year.fillna(df2["bdate"].dt.year.median()))

current_year = 2024
df2["age"] = current_year - df2["bdate_year"]
df2["is_employed"] = df2["career_end"] > df2["career_start"]
df2.drop(["id", "last_seen", "langs", "bdate",
          "career_end", "career_start"], axis=1, inplace=True)

def remove_False_life_main(row):
    if row["life_main"] not in "12345678910":
        return -1
    return int(row["life_main"])
def remove_False_people_main(row):
    if row["people_main"] not in "12345678910":
        return -1
    return int(row["people_main"])

df2["life_main"] = df2.apply(remove_False_life_main, axis=1)
df2["people_main"] = df2.apply(remove_False_people_main, axis=1)

df2 = pd.get_dummies(df2, columns=["education_form", "education_status", "occupation_type"], drop_first=True)


y_pred2 = classifier.predict(df2)
result = pd.DataFrame({'id': ID, 'result': y_pred2})
result.to_csv("result.csv", index=False)
