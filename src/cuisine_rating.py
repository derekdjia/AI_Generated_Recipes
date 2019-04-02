import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    """
    This function generates a confidence intervals graph of each cuisine
    """
    
    yum = pd.read_pickle('data/yummly_clean.pkl')

    cuisine_dic = {'Thai, Asian': 'Thai', 'Chinese, Asian':'Chinese', 'Japanese, Asian':'Japanese',
     'Southern & Soul Food, American': 'Southern & Soul Food',
     'Mediterranean, Greek': 'Mediterranean',
     'Cajun & Creole, Southern & Soul Food, American': 'Southern & Soul Food',
     'Asian, Japanese': 'Japanese','Cajun & Creole, American': 'Cajun & Creole',
     'Hawaiian, American': 'Hawaiian', 'Asian, Thai': 'Thai', 'American, Cuban':'Cuban',
     'Greek, Mediterranean': 'Greek', 'Indian, Asian': 'Indian','Asian, Chinese':'Chinese',
     'American, Kid-Friendly': 'American', 'Spanish, Portuguese':'Spanish',
     'Mexican, Southwestern': 'Mexican', 'Southwestern, Mexican': 'Southwestern',
     'American, Southern & Soul Food': 'Southern & Soul Food',
     'Cajun & Creole, Southern & Soul Food': 'Southern & Soul Food',
     'Portuguese, American':'American','American, French': 'American',
     'American, Cajun & Creole':'American',
     'American, Cajun & Creole, Southern & Soul Food': 'American',
     'Irish, American':'American'
        }

    yum['cuisine'] = yum['cuisine'].apply(lambda x: cuisine_dic[x] if x in cuisine_dic else x)
    df = yum[['cuisine', 'rating']]

    counts = df.groupby(['cuisine']).count().rename(index=str, columns={'rating':'count'}).reset_index()
    ratings = df.groupby(['cuisine']).mean().rename(index=str, columns={'rating':'avg_rating'}).reset_index()
    std_errs = df.groupby(['cuisine']).agg(np.std, ddof=1).rename(index=str, columns={'rating':'std_err'}).reset_index()

    final = pd.merge(counts,ratings)
    final = pd.merge(final,std_errs)
    final['avg_std_err'] = final['std_err']/np.sqrt(final['count']-1)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()
    plt.errorbar(final.cuisine, final.avg_rating, yerr=2*final.avg_std_err, fmt='o', color='black',
                 ecolor='lightgray', elinewidth=3, capsize=0)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    plt.savefig('img/cuisinerating.jpg',bbox_inches="tight")
