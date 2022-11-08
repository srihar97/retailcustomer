#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[36]:


#KMeans clustering
data_km = pd.read_csv(r"C:\Users\SRIHARI KARANAM\rfm.csv")
data_km.head()


# In[37]:


data_km.drop(['Unnamed: 0'], axis = 1, inplace = True)
data_km.head()


# In[38]:


data_km = data_km.iloc[:, :4]
data_km


# In[39]:


# Checking variation with different columns
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.scatter(data_km.recency, data_km.frequency, color='green', alpha=0.3)
plt.title('Recency vs Frequency', size=15)
plt.subplot(1,3,2)
plt.scatter(data_km.monetary, data_km.frequency, color='red', alpha=0.3)
plt.title('Monetary vs Frequency', size=15)
plt.subplot(1,3,3)
plt.scatter(data_km.recency, data_km.monetary, color='blue', alpha=0.3)
plt.title('Recency vs Monetary', size=15)
plt.show()


# In[40]:


# Checking outliers
column = ['recency', 'frequency', 'monetary']
for i in column:
    sns.boxplot(data_km[i]); plt.show() 


# In[41]:


#Outlier treatment with winsorization
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method ='iqr', # choose IQR rule boundaries or gaussian for mean and std
                   tail = 'both', # cap left, right or both tails
                   fold = 1.5,
                  # variables = ['']
                  )

for i in column:
    data_km[i] = winsor.fit_transform(data_km[[i]])


# In[42]:


bx_rfm = sns.boxplot(data = data_km,orient ="h", palette = "Set2" )


# In[43]:


# Removing retailer_id as it will not used in making cluster
data_km = data_km.iloc[:,1:]
data_km.columns


# In[44]:


## Standardization
from sklearn.preprocessing import StandardScaler
# scaling the variables and store it in different data
standard_scaler = StandardScaler()
data_sc = standard_scaler.fit_transform(data_km)

# converting it into dataframe
data_km_sc = pd.DataFrame(data_sc)
data_km_sc.columns = ['recency','frequency','monetary']
data_km_sc.head()


# In[45]:


###### scree plot or elbow curve ############
from sklearn.cluster import KMeans
TWSS = []  ## Total with sum of the squares.
k = list(range(2, 11))  ## range of clusters, k=2,3,4,5,6,7,8,9,10,11
## if use more than 10, its a meaning less because of we have only 25 universities.
## Think logically and decide how many clusters.
#for each value of i, k means clustering is done and  inertia is recorded
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data_km_sc)
    TWSS.append(kmeans.inertia_)  ## data filled in TWSS
    
TWSS ## for each of the k value, for 9 clusters have less records in each cluster
# and then get less WSS.


# In[46]:


# Scree plot/elbow curve
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# In[47]:


## finding Silhoutte score to destify the elbow curve k value.
from sklearn.metrics import silhouette_score
silhouette_avg_norm = []
for num_clusters in k:
    # initialise kmeans
    kmeans1 = KMeans(n_clusters =num_clusters)
    kmeans1.fit(data_km_sc)
    cluster_labels = kmeans1.labels_
    
    # silhouette score
    silhouette_avg_norm.append(silhouette_score(data_km_sc, cluster_labels))

plt.plot(k, silhouette_avg_norm, 'bx-') 
plt.xlabel('values of k') 
plt.ylabel ('silhouette score') 
plt.show()


# In[48]:


# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(data_km_sc)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(data_km_sc, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


# In[49]:


#Standardization from silhoutte score we got k = 3 


# In[50]:


#K-Means model
## Normalization of the data and silhoute_score conforms a k = 3

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)  # check n_clusters is 3
y = model.fit(data_km_sc)


# In[51]:


model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
data_km['clust'] = mb # creating a  new column and assigning it to new column 

data_km.head()


# In[52]:


# Rearrange the order of the columns.
data_km = data_km.iloc[:,[3,0,1,2]] ## rearranging the columns
data_km.head()


# In[53]:


data_km.iloc[:, 0:4].groupby(mb).mean() ## [rows; coulmn:coulmn]


# In[54]:


column = ['recency','frequency','monetary']
plt.figure(figsize=(15,4))
for i,j in enumerate(column):
    plt.subplot(1,3,i+1)
    sns.boxplot(y = data_km[j], x = data_km['clust'], palette = 'spring')
    plt.title('{} wrt clusters'.format(j.upper()), size=13)
    plt.ylabel('')
    plt.xlabel('')

plt.show()


# In[55]:


# Creating figure
fig = plt.figure(figsize = (8, 5))
ax = plt.axes(projection ="3d")

# Creating plot
ax.scatter3D(data_km.recency, data_km.frequency, data_km.monetary, c = data_km.clust, cmap='Accent')
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
plt.title('RFM in 3D with Clusters', size=15)
ax.set(facecolor='white')
plt.show()


# In[56]:


data_km['clust'].value_counts()


# In[57]:


#Classification DT(Decision Tree)
data_class = pd.read_csv(r"C:\Users\SRIHARI KARANAM\retail_km_3.csv")
data_class.shape


# In[58]:


data_class.drop(['Unnamed: 0'], axis = 1, inplace = True)
X = data_class.drop('clust', axis = 1)
Y = data_class.clust


# In[59]:


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0) # 70% training and 30% test


# In[60]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini", random_state=0,max_depth=5, min_samples_leaf=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)


# In[61]:


#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Test Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[62]:


y_pred_train = clf.predict(X_train)
# Model Accuracy, how often is the classifier correct?
print("Train Accuracy:",metrics.accuracy_score(y_train, y_pred_train))


# In[63]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay,RocCurveDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
clust_3 = accuracy_score(y_test, y_pred )

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, y_pred_train)}")
print(f"Test Accuracy of Decision Tree Classifier is {clust_3} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, y_pred)}\n")
print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")


# In[64]:


import pickle
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# In[65]:


clust_3 = 'credit_model'
pickle.dump(model,open(clust_3,'wb'))


# In[66]:


loaded_model=pickle.load(open(clust_3,'rb'))
loaded_model


# In[67]:


@st.cache(persist=True)
def prediction(recency, frequency, monetary):
    
    prediction= model.predict(pd.DataFrame([[recency, frequency, monetary]]))
    
    return prediction

def main():
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:black;text-align:center;"> Retailer Classification in Pharma </h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
    
    frequency = st.slider('frequency:', 1 , 180)
    monetary = st.slider('monetary:', 30 , 50000)
    recency = st.slider('recency:', 1 , 15)
    
    result = ""
    
    if st.button('classify'):
        result = prediction(recency, frequency, monetary)
        st.success(f'The retailer belongs to the cluster {result[0]:.0f}')
    
    
if __name__== '__main__':
    main()


# In[ ]:




