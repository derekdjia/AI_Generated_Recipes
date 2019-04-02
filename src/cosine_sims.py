import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

if __name__ == '__main__':
    """
    Constructs the graph of ingredient similarity
    """

    yum_ingr = pd.read_pickle('data/yummly_ingr.pkl')
    cuisines = yum_ingr['cuisine'].unique()
    df = yum_ingr.drop(['Unnamed: 0', 'id', 'recipeName', 'totalTimeInSeconds', 'course', 'ingredients', 'clean ingredients', 'len_diff',
                        'match ingredients', 'len_match'], axis=1).groupby('cuisine').mean()
    ingr_mat = df.copy().drop(['rating'], axis=1)    
    dist_out = 1-pairwise_distances(ingr_mat, metric="cosine")    
    
    mask = np.zeros_like(dist_out)
    mask[np.triu_indices_from(mask)] = True    
    mask2 = np.reshape([True if (i>0.4 and i<0.8) else False for i in dist_out.flatten()], (25,25))
    mask3 = mask+mask2
    mask = np.reshape([True if (i>0.5) else False for i in mask3.flatten()], (25,25))
    
    with sns.axes_style("white"):
        plt.figure(figsize=(18,18))
        sns.heatmap(dist_out, mask=mask, square=True, xticklabels=cuisines, yticklabels=cuisines)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.title('Cosine Similarities of Cuisines \n (Light = Most Similar, Dark = Most Dissimilar)', size = 28)
        plt.savefig('img/ingredientSimilarity.jpg')
