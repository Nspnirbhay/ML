#!/usr/bin/env python
# coding: utf-8

# # Task 1 : Prediction using Supervised ML

# To predict the percentage of marks of the students based on the number of hours they studied

# # Author : Nirbhay Kumar

# In[ ]:


import pandas as pd
import numpy as np 
from sklearn import linear_model
import matplotlib.pyplot as plt 


# In[23]:


data1 = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
data1


# In[24]:


data1.isnull == True


# In[25]:


data1.head()


# In[26]:


data1.tail()


# In[27]:


data1.describe()


# In[28]:


data1.shape


# In[29]:


data1.info()


# In[30]:


data1.corr()


# In[34]:


#matplotlib inline
plt.title('Scores Vs Hours')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.scatter(data1.Hours,data1.Scores,color='red',marker='+')


# In[15]:


plt.style.use('ggplot')
df.plot(kind='line')
plt.title('Scores Vs Hours')
plt.xlabel('Hours')
plt.ylabel('Scores')


# In[31]:


# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(data1[['Hours']],data1.Scores)


# In[32]:


new_df = data1.drop('Scores',axis='columns')
new_df


# In[33]:


Scores=data1.Scores
Scores


# In[35]:


#Regression

plt.title('Scores Vs Hours')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.scatter(data1.Hours,data1.Scores,color='red',marker='+')
plt.plot(data1.Hours,reg.predict(data1[['Hours']]),color='blue')


# In[36]:


#Evaluating Model

from sklearn import metrics
from sklearn.metrics import r2_score


# In[37]:


y_predict =reg.predict(new_df)
y_predict
print('Mean Absolute Error: {}',format(metrics.mean_absolute_error(y_predict,Scores)))
print("R2-score: %.2f" %r2_score(y_predict,Scores))


# In[38]:


#Predicting the Score with the given value of Hours

Hours=9.25
Predicted_Score = reg.predict([[Hours]])
Predicted_Score


# In[ ]:




