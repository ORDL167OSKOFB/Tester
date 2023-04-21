 
# import pandas as pd
# from scipy.spatial.distance import euclidean, mahalanobis, cosine   
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
# from scipy.stats import pearsonr
# def similarity_based_classification(data, Activity, secondaryActivity):
#     Activity_A = data[data["\"ACTIVITY\""] == Activity]
#     Activity_B = data[data["\"ACTIVITY\""] == secondaryActivity]
    
#     mean_A = Activity_A.mean()
#     mean_B = Activity_B.mean()
#     pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
#     return mean_A, mean_B
    
    
            
# def Similarity_metrics(A1, A2, A1Name, A2Name):
    
#     euclidean_dist = euclidean(A1, A2)
   


from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import mahalanobis
from scipy.stats import pearsonr    
from sklearn.preprocessing import StandardScaler
import numpy as np


def similarity_based_classification(data, activity1, activity2):
    
    data = data[data["\"ACTIVITY\""].isin([activity1, activity2])]
    
    X = data.drop("\"ACTIVITY\"", axis=1)
    y = data["\"ACTIVITY\""]
    
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scale)
    
    X_activity1 = X_pca[y == activity1]
    X_activity2 = X_pca[y == activity2]
    
    return X_activity1, X_activity2

def Similarity_metrics(A, B, Aname, Bname):
    
    mean_A = np.mean(A, axis=0)
    mean_B = np.mean(B, axis=0)
    print("Mean A::", mean_A)
    print("Mean B::", mean_B)
    cosine_sim = cosine_similarity(A, B)[0][0]
    euclidean_dist = euclidean_distances(A, B)[0][0]
    
    conv = np.cov(np.concatenate((A, B), axis=0),  rowvar=False)
    Mahalanobis_dist =mahalanobis(mean_A, mean_B, conv)
    
    # calculated the cosine compliment cosine to align with the principle of Euclidean + Mahalanobis:
    # "The higher the distance the more dissimilar the two activities are"
    
    inv_cosine = 1 - cosine_sim
    
    new_cosine = inv_cosine / (1-inv_cosine)
    
    avg_distance = (euclidean_dist + Mahalanobis_dist + new_cosine )/3
    return [cosine_sim, euclidean_dist, Mahalanobis_dist, avg_distance]


    
        
                
                