#!/usr/bin/env python
# coding: utf-8

# # heart disease prediction

# ### This project contains different machine learning algorithms to predict potential heart diseases in people. The algorithms included are k neighbour classification,Support vector classifier,Decision Tree classifier and Random forest classifier.The dataset is taken from kaggle.

# ## Important libraries used:   
# ### 1. numpy: To work with arrays  
# ### 2.pandas: To work with csv files and dataframes  
# ### 3.mathplotlib: To create charts using pyplot,define parameters using rcParams and color them with cm.rainbow   
# ### 4.warnings: To ignore all warnings which might be showing up in the notebook due to past/future depreciation of a feature.
# ### 5.train_test_split: to split the dataset into training and testing data
# ### 6.StandardScaler: To scale all the features,so that the machine learning model adapts better to the dataset

# ## importing libraries

# In[34]:


# basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')

#other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#machine learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ##  importing dataset

# In[35]:


data=pd.read_csv(r"C:\Users\hp\Desktop\DM\heart.csv")


# In[36]:


data.info()# information of dataset


# In[37]:


data.head(10)


# In[38]:


data.describe()#The method revealed that the range of each variable is different


# ## correlation matrix using pyplot and rcparams

# In[39]:


rcParams['figure.figsize']=20,14
plt.matshow(data.corr())
plt.yticks(np.arange(data.shape[1]),data.columns)
plt.xticks(np.arange(data.shape[1]),data.columns)
plt.colorbar()


# ## histogram

# In[40]:


data.hist()


# ## bar chart for class label

# In[41]:


rcParams['figure.figsize']=8,6
plt.bar(data['target'].unique(),data['target'].value_counts(),color=['red','green'])
plt.xticks([0,1])
plt.xlabel('Target classes')
plt.ylabel('count')
plt.title('count of each target class')


# ## data preprocessing

# In[42]:


data=pd.get_dummies(data,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])
standardScaler=StandardScaler()
columns_to_scale=['age','trestbps','chol','thalach','oldpeak']
data[columns_to_scale]=standardScaler.fit_transform(data[columns_to_scale])


# ## machine learning 
# ### in this 4 algorithms and their varied their various parameters and compared the final models.dataset is split into 67% training data and 33% testing data.

# In[43]:


y=data['target']
x=data.drop(['target'],axis=1)
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.33,random_state=0)


# ## K-neighbors classifier
# ### This classifier looks for the classes of K nearest neighbors of a given data point and based on the majority class,it assigns a class to this data point. However, the number of neighbors can be varied.

# In[44]:


knn_scores=[]
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(x_train,y_train)
    knn_scores.append(knn_classifier.score(x_test,y_test))


# ## line graph of the number of neighbors and the test score achieved in each case.

# In[45]:


plt.plot([k for k in range(1,21)],knn_scores,color = 'red')
for i in range(1,21):
    plt.text(i,knn_scores[i-1],(i,knn_scores[i-1]))
plt.xticks([i for i in range(1,21)])
plt.xlabel('number of neighbors (k)')
plt.ylabel('scores')
plt.title('K neighbors Classifier scores for different K values')


# ### output : maximum score of 87% when number of neighbors chosen is 8.

# ## Support Vector Classifier
# ### This classifier aims at forming a hyperplane that can separate the classes as much ads possible by adjusting thw distance between the data points and the hyperplane. There are several kernels based on which the hyperplane is decided. Kernels used here are : linear,poly,rbf and sigmoid.

# In[49]:


svc_scores= []
kernels = ['linear','poly','rbf','sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel= kernels[i])
    svc_classifier.fit(x_train,y_train)
    svc_scores.append(svc_classifier.score(x_test,y_test))


# ### bar graph of scores achieved by each kernel using rainbow method for coloring the graphs.

# In[50]:


colors = rainbow(np.linspace(0,1,len(kernels)))
plt.bar(kernels,svc_scores,color=colors)
for i in range(len(kernels)):
    plt.text(i,svc_scores[i],svc_scores[i])
plt.xlabel('kernel')
plt.ylabel('scores')


# ### linear kernel performed best of all the kernels with 83%

# ## Decision tree classifier 
# ### This classifier creates a decision tree based on which,it assigns the class values to each data point. Here, we can vary the maximum number of features to be considered while creating the model. In this project features from 1 to 30 are used.(the total features in the dataset after dummy columns were added.)

# In[51]:


dt_scores = []
for i in range(1,len(x.columns)+1):
    dt_classifier = DecisionTreeClassifier(max_features = i,random_state = 0)
    dt_classifier.fit(x_train,y_train)
    dt_scores.append(dt_classifier.score(x_test,y_test))


# In[53]:


plt.plot( [i for i in range(1,len(x.columns)+1)],dt_scores,color = 'blue')
for i in range(1,len(x.columns)+1):
    plt.text(i,dt_scores[i-1],(i,dt_scores[i-1]))
plt.xticks([i for i in range(1,len(x.columns)+1)])
plt.xlabel('Max features')
plt.ylabel('Scores')
plt.title('decision tree classifier scores for different number of maximum fetures')


# ### the maximum score is 79% and is achieved for maximum features being selected to be either 2, 4 or 18.

# ## Random Forest 
# ###  This classifier takes the concept of decision trees to the next level. It creates a forest of trees where each tree is formed by a random selection of features  from the total features. Here,we can vary the number of trees that will be used to predict the class. here the test scores are calculated over 10,100,200,500 and 1000 trees. 

# In[54]:


rf_scores =[]
estimators = [10,100,200,500,1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators =i,random_state = 0)
    rf_classifier.fit(x_train,y_train)
    rf_scores.append(rf_classifier.score(x_test,y_test))


# In[57]:


colors = rainbow(np.linspace(0,1,len(estimators)))
plt.bar([i for i in range(len(estimators))],rf_scores,color = colors,width =0.8)
for i in range(len(estimators)):
    plt.text(i,rf_scores[i],rf_scores[i])
plt.xticks(ticks = [i for i in range(len(estimators))],labels = [str(estimator) for estimator in estimators])
plt.xlabel('Number of estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different number of estimators')


# ### Maximum score of 84% is achieved for both forest with 100 and 500 trees

# In[ ]:




