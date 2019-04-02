import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def tsne_cluster_cuisine(df,sublist):
    """
    Input: df is a dataframe of explanatory variables, sublist is the list of cuisines you want to include in plot
    Function: Using sklearn's t-SNE method to visualize clustering of recipes using their recipes, colored by cuisine, generates plot
    Output: None
    """
    lenlist=[0]
    df_sub = df[df['cuisine']==sublist[0]]
    lenlist.append(df_sub.shape[0])
    for cuisine in sublist[1:]:
        temp = df[df['cuisine']==cuisine]
        df_sub = pd.concat([df_sub, temp],axis=0,ignore_index=True)
        lenlist.append(df_sub.shape[0])
    df_X = df_sub.drop(['cuisine','recipeName'],axis=1)
    dist = squareform(pdist(df_X, metric='cosine'))
    tsne = TSNE(metric='precomputed').fit_transform(dist)
    
    palette = sns.color_palette("hls", len(sublist))
    plt.figure(figsize=(10,10))
    for i,cuisine in enumerate(sublist):
        plt.scatter(tsne[lenlist[i]:lenlist[i+1],0],\
        tsne[lenlist[i]:lenlist[i+1],1],c=palette[i],label=sublist[i])
    plt.legend()

    return None

def run_kmeans(df,num):
    """
    Input: df is a dataframe of explanatory variables, num is the number of clusters 
    Function: Calculates silhouette score for k clusters
    Output: Returns the silhouette score
    """
    lenlist=[0]
    df_sub = df[df['cuisine']==sublist[0]]
    lenlist.append(df_sub.shape[0])
    for cuisine in sublist[1:]:
        temp = df[df['cuisine']==cuisine]
        df_sub = pd.concat([df_sub, temp],axis=0,ignore_index=True)
        lenlist.append(df_sub.shape[0])
    df_X = df_sub.drop(['cuisine','recipeName'],axis=1)
    kmeans = KMeans(n_clusters=num, random_state=0, n_jobs=-1).fit(df_X)
    return silhouette_score(df_X, kmeans.labels_)

def find_k(df, k):
    """
    Input: df is a dataframe of explanatory variables, k the maximum nuumber of clusters we want to visualize
    Function: Calculates a range of silhouette scores by calling the run_kmeans function
    Output: Returns a list of silhouette scores
    """
    sil_scores = []
    for i in range(2, k):
        scores = (run_kmeans(df, i))
        sil_scores.append(scores)     
    return sil_scores 

if __name__ == '__main__':
    yum_ingr = pd.read_pickle('data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('data/yummly_ingrX.pkl')

                            
    #Visualizing just the data of cuisines originating for 4 different subcontinents
    sublist = ['Italian','American','Chinese','Indian']
    wholelist = list(set(yum_ingr['cuisine']))
    df_ingr = yum_ingrX.copy()
    df_ingr['cuisine'] = yum_ingr['cuisine']
    df_ingr['recipeName'] = yum_ingr['recipeName']

    plt.figure(0)
    tsne_cluster_cuisine(df_ingr,sublist)
    plt.title('t-Distributed Stochastic Neighbor Embedding:\n 2D Visualization of Ingredients by Cuisine', size=20)
    plt.savefig('img/tsnecluster2.jpg')
    plt.figure(1)
    tsne_cluster_cuisine(df_ingr,wholelist)
    plt.title('t-Distributed Stochastic Neighbor Embedding:\n 2D Visualization of Ingredients by Cuisine', size=20)
    plt.savefig('img/tsnecluster3.jpg')

    sils = find_k(df_ingr, 20)

    plt.figure(2)
    plt.title("Silhouette Score vs Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.xlabel("Number of Clusters")
    plt.grid()
    x = range(2, len(sils)+2)
    plt.plot(x, sils)
    plt.savefig('img/silhouette.jpg')
    
