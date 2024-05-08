import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def ingest_data(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    #suppression des ligne avec valeur manquantes
    df = df.dropna(axis=0)
    #remplassmeent des valeurs non numérique
    df['sex'] = df['sex'].replace({'male' : 0, 'female' : 1})
    return df

def train_model(df: pd.DataFrame) -> ClassifierMixin:
    #instantiate model
    model = KNeighborsClassifier(4)
    #split data
    y= df['survived']
    x = df.drop( 'survived', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20, random_state=42)
    #train model
    model.fit(x_train,y_train)
    model.score (x_test,y_test)
    #evaluate model
    score = model.score(x_test, y_test)
    print(f"model score: {score}")
    return model 

if __name__ == "__main__":
    df = ingest_data("titanic.xls")
    df = clean_data(df)
    model = train_model(df)
    joblib.dumb( model, "model_titanic.joblib" )

