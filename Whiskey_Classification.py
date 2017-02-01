import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

whiskey = pd.read_csv("./whiskies.txt")
whiskey["Region"] = pd.read_csv("./regions.txt")
flavors = whiskey.iloc[:,2:14]
corr_flavors = pd.DataFrame.corr(flavors)

plt.figure(figsize=(10,10))
plt.pcolor(corr_flavors)
plt.colorbar()


corr_whiskeys = pd.DataFrame.corr(flavors.transpose())
plt.figure(figsize=(10,10))
plt.pcolor(corr_whiskeys)
plt.axis("tight")
plt.colorbar()

from sklearn.cluster.bicluster import SpectralCoclustering

model = SpectralCoclustering(n_clusters=6, random_state=0)
model.fit(corr_whiskeys)
model.rows_.shape
np.sum(model.rows_, axis=1)

whiskey["Group"] = pd.Series(model.row_labels_, index=whiskey.index)
whiskey = whiskey.ix[np.argsort(model.row_labels_)]
whiskey = whiskey.reset_index(drop=True)
correlations = pd.DataFrame.corr(whiskey.iloc[:, 2:14].transpose())
correlations = np.array(correlations)

plt.figure(figsize = (14,7))
plt.subplot(121)
plt.pcolor(corr_whiskeys)
plt.title("Original")
plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations)
plt.title("Rearranged")
plt.axis("tight")


data = pd.Series([1,2,3,4]) 
data = data.ix[[3,0,1,2]]

