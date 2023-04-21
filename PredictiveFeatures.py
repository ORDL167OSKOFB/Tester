import os

def feature_weights(X, model):
    feature_probabilities = []
    for feature in X.columns:
        X_mean = X.copy()
        X_mean[feature] = X[feature].mean()
        
        probability = model.predict_proba(X_mean)[:, 1]
        
        avg = probability.mean()
        feature_probabilities.append((feature, avg))
        feature_probabilities.append((feature, avg))
        
        
    return feature_probabilities
        
    
    
def print_top_features(feature_probabilities, n, A1):
    
    # Remove Duplicates 
    unique_dict = dict(feature_probabilities)
    unique_dict = list(unique_dict.items())
       
    top_features = sorted(unique_dict, key=lambda x: x[1], reverse=True)[:n]
    print(top_features)
    
    with open('Canvas_data\\top_features.txt', 'w') as f:
        for feature, probability in top_features:
            f.write(f"{feature}: {probability} \n")
            
    print(f"For {A1} the top {n} features that predict the probability of recognising an activity accurately are:")
    for feature, probability in top_features:
        print(f"{feature}: {probability}")
   
        
    return top_features
        