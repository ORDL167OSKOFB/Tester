
# Libraries 
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from PIL import Image

import csv
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
import os
import joblib
import seaborn as sns
import lime
import lime.lime_tabular
import shap 
import sys 
import io 
# Project Files 
# Executables\Activity_Similarities.py
import Activity_Similarities as Activity_Similarities
import pairwiseEuclidean as pairwiseEuclidean
import PDP as PDP
import ActivityDictionary as ActivityDictionary
import ActivityData as ActivityData
import PredictiveFeatures as PredictiveFeatures


activity_key = ActivityData.activity_dictionary()

dir = None
Activity = None
data = None
model = None
X = None
X_train = None
y_train = None
Activity_Comparisons = None
X_test = None


# Activity = sys.argv[1]


# Check if the model file exists, if not, fit the logistic regression model and save it

def Activity_init(ActivityInit):
    global Activity
    Activity = ActivityInit
    
# def dir_init(directory):
#     global dir
#     dir = dir
    
    
    
Activity_init('A')
print(Activity)
# dir_init(r"F:\Final Year UNI\Final Year Project Refactor\wisdm-dataset\arff_files\phone\accel")
dir = r"F:\Final Year UNI\Final Year Project Refactor\wisdm-dataset\arff_files\phone\accel"


# Check if the model file exists, if not, fit the logistic regression model and save it
if os.path.exists('CSV_files\data.csv'):
    print("File exists in proper path")
    model = joblib.load('CSV_files\model.pkl')
    data = pd.read_csv('CSV_files\data.csv')
    X = pd.read_csv('CSV_files\X_data.csv')
    X_train = pd.read_csv('CSV_files\X_train.csv')
    y_train = pd.read_csv('CSV_files\y_data.csv')
    X_test = pd.read_csv('CSV_files\X_test.csv')
    y = pd.read_csv('CSV_files\y_data.csv')
    
    y_test = pd.read_csv('CSV_files\y_test.csv')
    
    y_pred = pd.read_csv('CSV_files\y_pred.csv')
else:
        
    data = ActivityData.load_data_from_sensor(dir)
    data = ActivityData.preprocess_df(data)
    data.to_csv('CSV_files\data.csv', index=False)
    features = data.columns.tolist()
    # Calculating Accuracy
    X_train, X_test, y_train, y_test, X, y = ActivityData.split_data(Activity, data)
    X_test.to_csv('CSV_files\X_test.csv', index=False)
    X.to_csv('CSV_files\X_data.csv', index=False)
    X_train.to_csv('CSV_files\X_train.csv', index=False)
    y_train.to_csv('CSV_files\y_data.csv', index=False)
    y_test.to_csv('CSV_files\y_test.csv', index=False)
    
    y_pred, regressor = ActivityData.train(X_train, X_test, y_train)
    
    lreg = ActivityData.cross_validation(X, y, y_pred, regressor)
    model = joblib.dump(lreg, 'CSV_files\model.pkl')
    
    y_pred_df =  pd.DataFrame(y_pred, columns=['predicted_class'])
    y_pred_df.to_csv('CSV_files\y_pred.csv', index=False)
    



Activity_Similarity_file = f"CSV_files\Activity_Similarity-{Activity}.csv"

# If file exists load, otherwise generate similarity metrics
if os.path.exists(Activity_Similarity_file):
    Activity_Comparisons = pd.read_csv(Activity_Similarity_file, dtype={'Activity': str})
    
    
    Activity_heatmap = sns.heatmap(Activity_Comparisons.set_index('Activity'), annot=True, cmap="YlGnBu")
   
    Activity_heatmap.set_title("Distance metrics + Heatmap for Walking", fontweight = 'bold')
    Activity_heatmap.figure.savefig(f"Canvas_data\Activity_heatmap-A.png")
    # Activity_heatmap.figure.show(block = True)
    Activity_heatmap.figure.clf()
                    

    # print(Activity_Comparisons)
    # sns.heatmap(Activity_Comparisons.set_index('Activity'), annot=True, cmap="YlGnBu")
else:
    Other_Activities = ActivityDictionary.return_dict()
    save_file = f"CSV_files\Activity_Similarity-{Activity}.csv"
    metric_list = {}

    for secondaryActivity in Other_Activities:
        if Activity != secondaryActivity:
            A1Name = ActivityDictionary.get_activity_name(Activity)
            A2Name = ActivityDictionary.get_activity_name(secondaryActivity)
            A1, A2 = Activity_Similarities.similarity_based_classification(data, Activity, secondaryActivity)
            metrics = Activity_Similarities.Similarity_metrics(A1, A2, A1Name, A2Name)
            metric_list[A2Name] = metrics   

    with open(save_file, 'w', newline='') as f:
        writer = csv.writer(f)
        titles = ['Activity', 'Cosine Similarity', 'Euclidean', 'Mahalanobis', 'Average']
        writer.writerow(titles)
        for key, value in metric_list.items():
            writer.writerow([key] + value)

    Activity_Comparisons = pd.read_csv(f"CSV_files\Activity_Similarity-{Activity}.csv", dtype={'Activity': str})

    Activity_heatmap = sns.heatmap(Activity_Comparisons.set_index('Activity'), annot=True, cmap="YlGnBu")
   
    Activity_heatmap.set_title(f"Distance metrics + Heatmap for {A1Name}", fontweight = 'bold')
    Activity_heatmap.figure.savefig(f"Canvas_data\Activity_heatmap-{Activity}.png")
    # Activity_heatmap.figure.show(block = True)
    Activity_heatmap.figure.clf()
                    





# Euclidian Distance for Selected Activity 
print("We're at Euclidean Distance now")
dist_matrix = pairwiseEuclidean.distance_matrixCalc(Activity, data)
pairwiseEuclidean.plot(dist_matrix, Activity)










# print("We're at SHAP now")
# Define predict function for model
def predict_proba(X):
    return model.predict_proba(X)[:, 1]
# Initialize SHAP explainer and compute SHAP values
explainer = shap.Explainer(predict_proba, X)
shap_values = explainer(X)

# Create summary plot
shap_fig = shap.summary_plot(shap_values, X)

plt.savefig(f"Canvas_data\SHAP-{Activity}.png")








# Classification Report 

# y_pred, regressor = ActivityData.train(X_train, X_test, y_train)
y_pred = np.array(y_pred)

# resize y_pred to match the length of y
y_pred_resized = np.resize(y_pred, y.shape)

report = classification_report(y_test, y_pred)
print("Classification Report:" , report)

f1_score = f1_score(y, y_pred_resized, average='weighted')
report += f"\n\n F1: {f1_score:.3f}"

with open(f"Canvas_data\classification_report_{Activity}.txt", "w") as report_file:
    report_file.write(report)










data = ActivityData.load_data_from_sensor(dir)
data = ActivityData.preprocess_df(data)
data.to_csv('data.csv', index=False)
features = data.columns.tolist()
    # Calculating Accuracy
X_train, X_test, y_train, y_test, X, y = ActivityData.split_data(Activity, data)
feature_probabilities = PredictiveFeatures.feature_weights(X, model)
Activity_name = ActivityDictionary.get_activity_name(Activity)
top_features = PredictiveFeatures.print_top_features(feature_probabilities, 5,Activity_name)

feature_names = []
for tuple in top_features:
    feature_names.append(tuple[0])


explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=X.columns, class_names=['0', '1'], mode='classification')

# Choose a random instance to explain
idx = 0
instance = X.iloc[[idx]]
true_class = y.iloc[idx]

# Use the LIME explainer to generate an explanation for the instance

exp = explainer.explain_instance(instance.values[0], model.predict_proba, num_features=5)

# Print the true class of the instance and the predicted class probabilities
print('True class:', true_class)
print('Predicted probabilities:', model.predict_proba(instance))

# Print the top features contributing to the prediction
print('Top features:')

preds = model.predict_proba(instance)
with open(f"Canvas_data\LIME-{Activity}.txt", "w") as LIME_doc:
    LIME_doc.write(f"True class: {true_class} \n")
    LIME_doc.write(f'Predicted probabilities: {preds} \n')

    for feature, weight in exp.as_list():
        LIME_doc.write(f"{feature}, {weight} \n")









# Split PDP into rows of 5 for presentation

fig, axs = plt.subplots(5, 5, figsize=(20,20))

for i in range(5):
    for j in range(5):
        feature = X.columns[i*5+j]
        grid = np.linspace(X[feature].min(), X[feature].max(), num=len(X))
        X_grid = X.copy()
        X_grid[feature] = grid
        probas = model.predict_proba(X_grid)[:, 1]
        avg_probas = np.array([probas[X_grid[feature] == val].mean() for val in grid])
        
        axs[i, j].plot(grid, avg_probas)
        axs[i, j].set_xlabel(feature)
        axs[i, j].set_ylabel(f'Predicted probability of {Activity_name}')
        print("iter", i, ":", j)
        
fig.suptitle('Partial Dependence Plot for Each Feature')
plt.tight_layout()
plt.savefig('PDP{}.png'.format(Activity))
plt.show()
plt.close()   


















