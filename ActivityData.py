import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from scipy.io import arff 
from IPython.display import display
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import pdist, squareform

from sklearn.model_selection import KFold

def map_activity(df, mapping):
    
    key_column = df["\"ACTIVITY\""]
    mapping = key_column.replace(mapping)
    df["\"ACTIVITY\""] = mapping
    return df


def activity_dictionary():
    
    mapping = {}

    with open(r"F:\Final Year UNI\Final Year Project Refactor\wisdm-dataset\activity_key.txt", 'r') as f:
     for line in f.readlines()[::-1]:
        value, key = line.strip().split(' = ')
        activity_name = key.strip()  # extract the activity name from the key-value pair
        mapping[activity_name] = value.strip()

    
    return mapping


def load_data_from_sensor(directory):
    
    files = os.listdir(directory)
    df_list = pd.DataFrame()
    # Iterate through the list of files
    for file in files:
        if file.endswith('.arff'):
            f = os.path.join(directory, file)
            if (os.path.isfile(f)):
                arf = arff.loadarff(f)
                df = pd.DataFrame(arf[0])
                df["\"ACTIVITY\""] = df["\"ACTIVITY\""].str.decode('UTF-8')
                df_list = pd.concat([df_list, df], ignore_index=True)
    return df_list


def preprocess_df(df_list):
    
    # Remove single point axis values
    df_list = df_list.drop(["\"X0\"", "\"X1\"", "\"X2\"","\"X3\"","\"X4\"","\"X5\"","\"X6\"","\"X7\"","\"X8\"","\"X9\"",], axis=1)
    df_list = df_list.drop(["\"Y0\"", "\"Y1\"", "\"Y2\"","\"Y3\"","\"Y4\"","\"Y5\"","\"Y6\"","\"Y7\"","\"Y8\"","\"Y9\"",], axis=1)
    df_list = df_list.drop(["\"Z0\"","\"Z1\"", "\"Z2\"","\"Z3\"","\"Z4\"","\"Z5\"","\"Z6\"","\"Z7\"","\"Z8\"","\"Z9\"",], axis=1)

    # Take a subset of MFCCs for each axis 
    df_XMFCC = df_list[["\"XMFCC0\"", "\"XMFCC1\"","\"XMFCC2\"","\"XMFCC3\"","\"XMFCC4\"","\"XMFCC5\"","\"XMFCC6\"","\"XMFCC7\"","\"XMFCC8\"","\"XMFCC9\"","\"XMFCC10\"","\"XMFCC11\"","\"XMFCC12\""]]
    df_YMFCC = df_list[["\"YMFCC0\"", "\"YMFCC1\"","\"YMFCC2\"","\"YMFCC3\"","\"YMFCC4\"","\"YMFCC5\"","\"YMFCC6\"","\"YMFCC7\"","\"YMFCC8\"","\"YMFCC9\"","\"YMFCC10\"","\"YMFCC11\"","\"YMFCC12\""]]
    df_ZMFCC = df_list[["\"ZMFCC0\"", "\"ZMFCC1\"","\"ZMFCC2\"","\"ZMFCC3\"","\"ZMFCC4\"","\"ZMFCC5\"","\"ZMFCC6\"","\"ZMFCC7\"","\"ZMFCC8\"","\"ZMFCC9\"","\"ZMFCC10\"","\"ZMFCC11\"","\"ZMFCC12\""]]


    # Calculate Average per axis 
    df_list.loc[:,'mean_XMFCC'] = df_XMFCC.mean(axis=1)
    df_list.loc[:,'mean_YMFCC'] = df_YMFCC.mean(axis=1)
    df_list.loc[:,'mean_ZMFCC'] = df_ZMFCC.mean(axis=1)

    # Drop all single point MFCC values per axis
    df_list = df_list.drop(["\"XMFCC0\"", "\"XMFCC1\"","\"XMFCC2\"","\"XMFCC3\"","\"XMFCC4\"","\"XMFCC5\"","\"XMFCC6\"","\"XMFCC7\"","\"XMFCC8\"","\"XMFCC9\"","\"XMFCC10\"",
                        "\"XMFCC11\"","\"XMFCC12\"", 
                        "\"YMFCC0\"", "\"YMFCC1\"","\"YMFCC2\"","\"YMFCC3\"","\"YMFCC4\"","\"YMFCC5\"","\"YMFCC6\"","\"YMFCC7\"","\"YMFCC8\"","\"YMFCC9\"","\"YMFCC10\"",
                        "\"YMFCC11\"","\"YMFCC12\"", 
                        "\"ZMFCC0\"", "\"ZMFCC1\"","\"ZMFCC2\"","\"ZMFCC3\"","\"ZMFCC4\"","\"ZMFCC5\"","\"ZMFCC6\"","\"ZMFCC7\"","\"ZMFCC8\"","\"ZMFCC9\"","\"ZMFCC10\"",
                        "\"ZMFCC11\"","\"ZMFCC12\""], axis =1)


  
    
    # Drop class column (specifies subject. Reason? Activity recognition should not have dependencies on subjects,
    # or this would be "activity recognition for derek")
    df_list = df_list.drop(["\"class\""], axis =1)
    
    
    return df_list

# Passes dataframe + Activity 
 
    
def split_data(activity, df_list):
    
    
    X = df_list.drop("\"ACTIVITY\"", axis=1)
    # Line may not work
    y = (df_list["\"ACTIVITY\""] == activity).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X, y

def train(X_train, X_test, y_train):
    regressor = LogisticRegression(max_iter = 23074)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    return y_pred, regressor

    
def cross_validation(X,y, y_pred, regressor):
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)

    # calculate the average accuracy score
    mean_score = sum(scores) / len(scores)

    # print the average accuracy score
    print(f"Average accuracy score: {mean_score:.3f}") 
    
    return regressor  
    

if __name__ == "__main__":
    # Script code
    pass