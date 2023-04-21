from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import matplotlib.pyplot as plt2


def distance_matrixCalc(activity, df):
    
    activity_scaled = df.loc[df['"ACTIVITY"'] == activity].drop('"ACTIVITY"', axis=1)
    scaler = MinMaxScaler()
    activity_scaled = scaler.fit_transform(activity_scaled)
    distances = pdist(activity_scaled , metric='euclidean')
    distance_matrix = squareform(distances)
    return distance_matrix

def plot(distance_matrix, Activity):
    # print(distance_matrix.shape)
    # print(distance_matrix)

# create a histogram
    plt2.title('Euclidean Distance Matrix for Activity A')

    plt2.hist(distance_matrix, bins=5)
    plt2.savefig('Canvas_data\Pairwise_Euclidean-{}.png'.format(Activity))
    plt2.close()
   

    

