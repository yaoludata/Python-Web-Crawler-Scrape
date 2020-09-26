#!/usr/bin/env python
# coding: utf-8

# In[1]:


# connect to google drive
from google.colab import drive
drive.mount('/content/drive')


# In[2]:


# import dataset
import pandas as pd

invoice = pd.read_csv('---------')


# In[3]:


# check the dataset
invoice.head()


# In[ ]:


# create category column (boolean) : Indicator of whether invoice is under dispute or not
invoice['category'] = invoice['disputed_at'].notnull()


# In[6]:


# have a look the dataset
invoice.head()


# In[ ]:


# import datetime library
from datetime import datetime


# In[ ]:


# transform date/ due_date / paid_date columns to datetime type
def autoconvert_datetime(value):
    formats = ['%Y-%m-%d %H:%M:%S']  # formats to try
    result_format = '%Y-%m-%d'  # output format
    for dt_format in formats:
        try:
            dt_obj = datetime.strptime(value, dt_format)
            return dt_obj.strftime(result_format)
        except Exception as e:  # throws exception when format doesn't match
            pass
    return value  # let it be if it doesn't match


# In[ ]:


# transform date columns to datetime type
invoice['date'] = invoice['date'].apply(autoconvert_datetime)


# In[ ]:


# transform due_date columns to datetime type
invoice['due_date'] = invoice['due_date'].apply(autoconvert_datetime)


# In[ ]:


# transform paid_date columns to datetime type
invoice['paid_date'] = invoice['paid_date'].apply(autoconvert_datetime)


# In[ ]:


# transform date columns to datetime type
invoice['date'] = pd.to_datetime(invoice['date'], format='%Y-%m-%d', errors='coerce')


# In[ ]:


# transform paid_date columns to datetime type
invoice['paid_date'] = pd.to_datetime(invoice['paid_date'], format='%Y-%m-%d', errors='coerce')


# In[ ]:


# transform due_date columns to datetime type
invoice['due_date'] =  pd.to_datetime(invoice['due_date'], format='%Y-%m-%d', errors='coerce')


# In[13]:


# check missing values
print(invoice['date'].isnull().sum())
print(invoice['due_date'].isnull().sum())
print(invoice['paid_date'].isnull().sum())


# In[ ]:


# drop missing values
invoice = invoice.dropna(subset=['date', 'paid_date', 'due_date'])


# In[15]:


# double check missing values
print(invoice['date'].isnull().sum())
print(invoice['due_date'].isnull().sum())
print(invoice['paid_date'].isnull().sum())


# In[ ]:


# create late_days columns: the number of days for invoice payment
invoice.insert(8, 'late_days', invoice['paid_date'] - invoice['due_date'])


# In[17]:


# check dataset after data preprocessing
invoice.head()


# In[ ]:


# transform the data type of late_days columns to int
invoice['late_days'] = invoice['late_days'].dt.days


# In[19]:


# check missing values
print(invoice['due_date'].isnull().sum())
print(invoice['paid_date'].isnull().sum())
print(invoice['late_days'].isnull().sum())


# In[20]:


# check dataset after data preprocessing
invoice.head()


# In[ ]:


# create label columns
invoice.insert(9, 'label', invoice.late_days)


# In[22]:


# check dataset after data preprocessing
invoice.head()


# In[ ]:


# set the label to each invoice: (0: on time / 1: 1-30days / 2: 31-60days / 3: 61-90days / 4: 90+ days) 
invoice.loc[invoice.late_days <= 0, 'label']  = 0
invoice.loc[(invoice.late_days > 0) & (invoice.late_days < 31), 'label'] = 1
invoice.loc[(invoice.late_days > 30) & (invoice.late_days < 61), 'label']  = 2
invoice.loc[(invoice.late_days > 60) & (invoice.late_days < 91), 'label']  = 3
invoice.loc[invoice.late_days > 90, 'label']  = 4


# In[24]:


# check dataset after data preprocessing
invoice.head()


# In[25]:


# check the distribution of each label
invoice_count = invoice.groupby('label').count()
invoice_count


# In[26]:


# sort dataset by date
invoice.sort_values(by=['date'])


# In[28]:


# sort dataset by late_days
invoice.sort_values(by=['late_days'])


# In[29]:


# check the information of dataset 
invoice.info()


# In[30]:


# check missing values
invoice.isnull().sum()


# In[31]:


# statistics summary of late_days
invoice['late_days'].describe().apply(lambda x: format(x, 'f'))


# In[32]:


# visualize the invoice payment distribution (All invoices until 2020/04/07)
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

labels = ['On Time', '1-30 Days', '31-60 Days', '61-90 Days', '91+ Days']
plt.xticks(invoice_count.index, labels, rotation=50)
plt.xlabel('Payment Classes')
plt.ylabel('Number of Invoices')
plt.title('Invoice Payment Distribution Until 2020/04/07')
plt.bar(invoice_count.index, invoice_count.id)
plt.grid(color='#95a5a6', linestyle='--', linewidth=0.5, axis='y', alpha=0.7)
plt.ticklabel_format(style='plain', axis='y')
plt.show()


# In[ ]:


# create 'payment_term' column: The deadline of payment due
invoice.insert(8, 'payment_term', invoice['due_date'] - invoice['date'])


# In[ ]:


# transform payment_term to int
invoice['payment_term'] = invoice['payment_term'].dt.days


# In[35]:


# check the dataset
invoice.head()


# In[ ]:


# rename column
invoice.rename(columns={'total_amount_cents':'invoice_base_amount'}, inplace = True)


# In[37]:


invoice.head()


# In[38]:


# check dataset information
invoice.info()


# In[ ]:


# selete the data from 2019/4/9 - 2020/4/9
start_date = '2019-04-09'
end_date = '2020-04-09'


# In[ ]:


# selete the data from 2019/4/9 - 2020/4/9
selected_date = (invoice['date'] >= start_date) & (invoice['date'] <= end_date)


# In[41]:


# selete the data from 2019/4/9 - 2020/4/9
invoice_df = invoice.loc[selected_date]
invoice_df


# In[ ]:


# sort the values by date 
invoice_df = invoice_df.sort_values(by=['date'])


# In[43]:


# check the dataset
invoice_df.head()


# In[ ]:


# create the 'num_paid_invoices' column: Number of paid invoices prior to the creation date of a new invoice of a customer
num_paid_invoices = invoice_df.groupby('contact_id').count()


# In[45]:


# rename the column
num_paid_invoices.rename(columns={'id':'num_paid_invoices'}, inplace = True)
num_paid_invoices


# In[ ]:


# left join the two dateset by contact_id
invoice_df = pd.merge(invoice_df,
                 num_paid_invoices['num_paid_invoices'],
                 on='contact_id', 
                 how='left')


# In[47]:


# check the dataset
invoice_df


# In[ ]:


# create 'num_paid_late_invoices' column: Number of invoices which were paid late prior to the creation date of a new invoice of a customer


# In[ ]:


paid_late = invoice_df['label'] >= 1


# In[50]:


invoice_late_df = invoice_df.loc[paid_late]
invoice_late_df


# In[51]:


paid_late_df = invoice_late_df.groupby('contact_id').count()
paid_late_df


# In[ ]:


# rename the column
paid_late_df.rename(columns={'id':'num_paid_late_invoices'}, inplace = True)


# In[53]:


# check the dataset
paid_late_df.head()


# In[ ]:


# left join the two dateset by contact_id
invoice_df = pd.merge(invoice_df,
                 paid_late_df['num_paid_late_invoices'],
                 on='contact_id', 
                 how='left')


# In[61]:


invoice_df.head()


# In[ ]:


# create 'ratio_of_paid_late_invoices' column: Number of invoices that were paid late / Number of paid invoices prior to the creation date of a new invoice of a customer.
invoice_df['ratio_of_paid_late_invoices'] = invoice_df['num_paid_late_invoices'] / invoice_df['num_paid_invoices']


# In[56]:


invoice_df.head()


# In[57]:


# create 'total_amount_paid' column: The sum of the base amount from all the paid invoices prior to a new invoice for a customer
total_amount_paid_df = invoice_df.groupby('contact_id').sum()
total_amount_paid_df 


# In[ ]:


# rename the column
total_amount_paid_df.rename(columns={'invoice_base_amount':'total_amount_paid'}, inplace = True)


# In[ ]:


# left join the two dateset by contact_id
invoice_df = pd.merge(invoice_df,
                 total_amount_paid_df['total_amount_paid'],
                 on='contact_id', 
                 how='left')


# In[60]:


# check the dataset
invoice_df.head()


# In[61]:


# create 'invoice_base_amount' column: The sum of the base amount from all the paid invoices prior to a new invoice for a customer
total_amount_paid_late_df = invoice_late_df.groupby('contact_id').sum()
total_amount_paid_late_df.head()


# In[ ]:


# rename the column
total_amount_paid_late_df.rename(columns={'invoice_base_amount':'total_amount_paid_late'}, inplace = True)


# In[63]:


# left join the two dataset by contact_id
invoice_df = pd.merge(invoice_df,
                 total_amount_paid_late_df['total_amount_paid_late'],
                 on='contact_id', 
                 how='left')
invoice_df


# In[ ]:


# create 'ratio_of_paid_late_amount' column: Ratio of sum of paid base amount that were late
invoice_df['ratio_of_paid_late_amount'] = invoice_df['total_amount_paid_late'] / invoice_df['total_amount_paid']


# In[65]:


# create 'total_avg_late_days' column: Average days late of all paid invoices that were late prior to a new invoice for a customer
avg_day_late_paid_invoices = invoice_late_df.groupby('contact_id').mean()
avg_day_late_paid_invoices.head()


# In[ ]:


# rename the column
avg_day_late_paid_invoices.rename(columns={'late_days':'total_avg_late_days'}, inplace = True)


# In[67]:


# left join the two datasets by contact_id
invoice_df = pd.merge(invoice_df,
                 avg_day_late_paid_invoices['total_avg_late_days'],
                 on='contact_id', 
                 how='left')
invoice_df 


# In[ ]:


# import ourstanding invoices dataset (Outstanding invoices are those that the company has yet to pay)
outstanding = pd.read_csv('/content/drive/My Drive/datascience/outstanding_true.csv')


# In[69]:


# check the dataset
outstanding


# In[70]:


# check the information of dataset
outstanding.info()


# In[ ]:


# transform datatype of date / due_date column to datetime
from datetime import datetime

outstanding['date'] = outstanding['date'].apply(autoconvert_datetime)
outstanding['due_date'] = outstanding['due_date'].apply(autoconvert_datetime)

outstanding['date'] = pd.to_datetime(outstanding['date'], format='%Y-%m-%d', errors='coerce')
outstanding['due_date'] =  pd.to_datetime(outstanding['due_date'], format='%Y-%m-%d', errors='coerce')


# In[72]:


# check the dataset
outstanding.head()


# In[ ]:


# copy the dataset
outstanding_df = outstanding.copy()


# In[74]:


# select the data from 2019/4/9 - 2020/4/9
start_date = '2019-04-09'
end_date = '2020-04-09'
selected_date_1 = (outstanding_df['date'] >= start_date) & (outstanding_df['date'] <= end_date)
outstanding_df = outstanding_df.loc[selected_date_1]
outstanding_df


# In[75]:


# sort values by date
outstanding_df = outstanding_df.sort_values(by=['date'])
outstanding_df


# In[76]:


# Some invoices are partially paid
outstanding_df.loc[outstanding_df['amount_paid_cents'] != 0].sort_values(by=['amount_paid_cents'])


# In[ ]:


# create num_outstanding_invoices column: Number of the outstanding invoices prior to the creation date of a new invoice of a customer.
num_outstanding_invoices = outstanding_df.groupby('contact_id').count()


# In[ ]:


# rename the coulumn
num_outstanding_invoices.rename(columns={'id':'num_outstanding_invoices'}, inplace = True)


# In[79]:


# check the dataset
num_outstanding_invoices


# In[80]:


# left join the two datasets by contact_id
invoice_df = pd.merge(invoice_df,
                 num_outstanding_invoices['num_outstanding_invoices'],
                 on='contact_id', 
                 how='left')
invoice_df


# In[ ]:


# create num_outstanding_late_invoices column: Number of the outstanding invoices which were late prior to the creation date of a new invoice of a customer
# we set the late date is 2020/4/21
late_date = '2020-04-21'
outstanding_late = outstanding_df['due_date'] < late_date


# In[82]:


outstanding_late_df = outstanding_df.loc[outstanding_late]
outstanding_late_df


# In[ ]:


num_outstanding_late_invoices = outstanding_late_df.groupby('contact_id').count()


# In[85]:


# rename the coulumn
num_outstanding_late_invoices.rename(columns={'id':'num_outstanding_late_invoices'}, inplace = True)
num_outstanding_late_invoices


# In[86]:


# left join the two datasets by contact_id
invoice_df = pd.merge(invoice_df,
                 num_outstanding_late_invoices['num_outstanding_late_invoices'],
                 on='contact_id', 
                 how='left')
invoice_df


# In[ ]:


# create ratio_outstanding_late_invoice coumn
#Ratio of outstanding invoices that were late
#num_outstanding_late_invoices / num_outstanding_invoices
invoice_df['ratio_outstanding_late_invoice'] = invoice_df['num_outstanding_late_invoices'] / invoice_df['num_outstanding_invoices']


# In[ ]:


# create amount_outstanding_invoices column: The sum of the base amount from all the outstanding invoices prior to a new invoice for a customer
amount_outstanding_invoices = outstanding_df.groupby('contact_id').sum()


# In[89]:


amount_outstanding_invoices.rename(columns={'amount_due_cents':'amount_outstanding_invoices'}, inplace = True)
amount_outstanding_invoices


# In[90]:


invoice_df = pd.merge(invoice_df,
                 amount_outstanding_invoices['amount_outstanding_invoices'],
                 on='contact_id', 
                 how='left')
invoice_df


# In[ ]:


# create amount_outstanding_late_invoices coulmn:The sum of the base amount from all the outstanding invoices which were late prior to a new invoice for a customer.
amount_outstanding_late_invoices = outstanding_late_df.groupby('contact_id').sum()


# In[92]:


amount_outstanding_late_invoices.rename(columns={'amount_due_cents':'amount_outstanding_late_invoices'}, inplace = True)
amount_outstanding_late_invoices


# In[93]:


invoice_df = pd.merge(invoice_df,
                 amount_outstanding_late_invoices['amount_outstanding_late_invoices'],
                 on='contact_id', 
                 how='left')
invoice_df


# In[ ]:


# create ratio_amount_outstanding_late_invoice column
# Ratio of sum of outstanding base amount that were late
# amount_outstanding_late_invoices/amount_outstanding_invoices
invoice_df['ratio_amount_outstanding_late_invoice'] = invoice_df['amount_outstanding_late_invoices'] / invoice_df['amount_outstanding_invoices']


# In[95]:


# late date is 2020-04-21
# Average days late of outstanding invoices being late.
outstanding_late_df['paid_date'] = '2020-04-21'


# In[ ]:


#outstanding_late_df.insert(12, 'late_days', outstanding_late_df['paid_date'] - outstanding_late_df['due_date'])
#outstanding_late_df['late_days'] = outstanding_late_df['late_days'].dt.days


# In[ ]:


# transform the date type
def autoconvert_datetime(value):
    formats = ['%Y-%m-%d']  # formats to try
    result_format = '%Y-%m-%d'  # output format
    for dt_format in formats:
        try:
            dt_obj = datetime.strptime(value, dt_format)
            return dt_obj.strftime(result_format)
        except Exception as e:  # throws exception when format doesn't match
            pass
    return value  # let it be if it doesn't match


# In[98]:


outstanding_late_df['paid_date'] = outstanding_late_df['paid_date'].apply(autoconvert_datetime)
outstanding_late_df['paid_date'] =  pd.to_datetime(outstanding_late_df['paid_date'], format='%Y-%m-%d', errors='coerce')


# In[99]:


# calculate the number of days which are late for each invoice
outstanding_late_df.insert(12, 'late_days', outstanding_late_df['paid_date'] - outstanding_late_df['due_date'])
outstanding_late_df['late_days'] = outstanding_late_df['late_days'].dt.days


# In[100]:


outstanding_late_df


# In[ ]:


# create avg_outstanding_late_days column: Average days late of outstanding invoices being late.
avg_outstanding_late_days = outstanding_late_df.groupby('contact_id').mean()


# In[102]:


avg_outstanding_late_days.rename(columns={'late_days':'avg_outstanding_late_days'}, inplace = True)
avg_outstanding_late_days


# In[ ]:


invoice_df = pd.merge(invoice_df,
                 avg_outstanding_late_days['avg_outstanding_late_days'],
                 on='contact_id', 
                 how='left')


# In[104]:


invoice_df


# In[ ]:


# drop the unused columns
invoice_final_df = invoice_df.drop(columns="organisation_id")
invoice_final_df = invoice_final_df.drop(columns="due_date")
invoice_final_df = invoice_final_df.drop(columns="paid_date")
invoice_final_df = invoice_final_df.drop(columns="disputed_at")
invoice_final_df = invoice_final_df.drop(columns="late_days")


# In[106]:


invoice_final_df 


# In[107]:


invoice_final_df.info()


# In[ ]:


# filter the first time customer
aaa1 = invoice_final_df.groupby('contact_id').count()


# In[109]:


aaa1.sort_values(by=['id'])


# In[ ]:


# 0: first time custormer / 1: returning customer
aaa1.loc[aaa1.id <= 1, 'label']  = 0
aaa1.loc[aaa1.id > 1, 'label']  = 1


# In[111]:


aaa1.rename(columns={'label':'customer'}, inplace = True)
aaa1


# In[ ]:


customer_count = aaa1.groupby('customer').count()


# In[ ]:


customer_count['sum'] = customer_count['id'].sum()


# In[ ]:


customer_count['size'] = customer_count['id'] / customer_count['sum'] * 100


# In[115]:


# check the distribution of first time customer and returning customer
customer_count


# In[ ]:


# import library
import matplotlib.pyplot as plt


# In[ ]:


a = customer_count['size']


# In[118]:


# visualize the distribution of first time customer and returning customer
labels = ['fist-time customer', 'returning customer']
sizes = a
explode = (0, 0.1)
colors = ['#ff9999','#66b3ff']

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.show()


# In[ ]:


# create a new dataset without first-time customer
invoice_final_df = pd.merge(invoice_final_df,
                 aaa1['customer'],
                 on='contact_id', 
                 how='left')


# In[ ]:


# delete the first time customer data. (we use returning customer data to create the model)
invoice_final_df = invoice_final_df[invoice_final_df.customer != 0]


# In[ ]:


invoice_final_df = invoice_final_df.drop('customer', axis=1)


# In[122]:


invoice_final_df


# In[ ]:


# check the distribution of each class
distribution = invoice_final_df.groupby('label').count()


# In[124]:


distribution


# In[125]:


# distribution of invoices payment after data processing (2019/4/7 - 2020/4/7)
labels = ['On Time', '1-30 Days', '31-60 Days', '61-90 Days', '91+ Days']
plt.xticks(distribution.index, labels, rotation=50)
plt.xlabel('Payment Classes')
plt.ylabel('Number of Invoices')
plt.title('Invoice Payment Distribution (2019/4/7 - 2020/4/7)')
plt.bar(distribution.index, distribution.id)
plt.grid(color='#95a5a6', linestyle='--', linewidth=0.5, axis='y', alpha=0.7)
plt.ticklabel_format(style='plain', axis='y')
plt.show()

print("                                                                                                               ")
print("                                                                                                               ")
print("                                                                                                               ")
print("                                                                                                               ")



distribution['sum'] = distribution['id'].sum()
distribution['size'] = distribution['id'] / distribution['sum'] * 100
b = distribution['size']
#colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#66b3ff']
sizes = b
explode = (0, 0.1, 0, 1, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, pctdistance=0.85, rotatelabels =True)

#draw circle
#centre_circle = plt.Circle((0,0),0.70,fc='white')
#fig = plt.gcf()
#fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
theme = plt.get_cmap('jet')
plt.show()


# In[126]:


# check the missing value
invoice_final_df.isnull().sum()


# In[ ]:


# fill the missing values with 0
invoice_final_df = invoice_final_df.fillna(0)


# In[129]:


# double check the missing values
invoice_final_df.isnull().sum()


# In[130]:


# check the dataset
invoice_final_df


# In[ ]:


# import library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


# import library
from sklearn.model_selection import train_test_split


# In[ ]:


# import library
from sklearn.model_selection import KFold


# In[134]:


# import library
import pandas as pd # to import csv and for data manipulation
import matplotlib.pyplot as plt # to plot graph
import seaborn as sns # for intractve graphs
import numpy as np # for linear algebra
import datetime # to dela with date and time
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split # to split the data
#from sklearn.cross_validation import KFold # For cross vbalidation
from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
import warnings
warnings.filterwarnings('ignore')


# In[135]:


# check the infomation of dataset
invoice_final_df.info()


# In[136]:


# now let us check in the number of Percentage for each class
on_time = len(invoice_final_df[invoice_final_df["label"]==0]) 
i_1_30_days = len(invoice_final_df[invoice_final_df["label"]==1])
i_31_60_days = len(invoice_final_df[invoice_final_df["label"]==2]) 
i_61_90_days = len(invoice_final_df[invoice_final_df["label"]==3]) 
more_than_90_days = len(invoice_final_df[invoice_final_df["label"]==4])  
all_total = on_time +i_1_30_days+i_31_60_days+i_61_90_days+more_than_90_days
Percentage_of_on_time = on_time/all_total
print("percentage of On Time Invoices",Percentage_of_on_time*100)
Percentage_of_i_1_30_days = i_1_30_days/all_total
print("percentage of 1 - 30 Days Invoices",Percentage_of_i_1_30_days*100)
Percentage_of_i_31_60_days = i_31_60_days/all_total
print("percentage of 31 - 60 Days Invoices",Percentage_of_i_31_60_days*100)
Percentage_of_i_61_90_days = i_61_90_days/all_total
print("percentage of 61 - 90 Days Invoices",Percentage_of_i_61_90_days*100)
Percentage_of_more_than_90_days = more_than_90_days/all_total
print("percentage of 90+ Days Invoices",Percentage_of_more_than_90_days*100)


# In[ ]:


# since our data is not balance, we should balance our data


# In[ ]:


invoice_model = invoice_final_df.copy()


# In[139]:


invoice_model.info()


# In[ ]:


# drop the unuesd columns
invoice_model = invoice_model.drop(columns="id")
invoice_model = invoice_model.drop(columns="date")
invoice_model = invoice_model.drop(columns="contact_id")


# In[141]:


invoice_model.info()


# In[ ]:


# Re-order Columns, and rename the column
invoice_model.insert(0, 'class', invoice_model.label)


# In[ ]:


invoice_model = invoice_model.drop(columns="label")


# In[144]:


invoice_model.info()


# In[145]:


invoice_model


# In[146]:


# transform currency_code to digit factor
# 0: NZD/ 1: AUD
invoice_model.loc[invoice_model.currency_code == 'NZD', 'currency_code']  = 0
invoice_model.loc[invoice_model.currency_code == 'AUD', 'currency_code']  = 1
invoice_model


# In[147]:


# check the distribution of each class
invoice_model.groupby('class').count()


# In[148]:


invoice_model.info()


# In[ ]:


# import library
from collections import Counter

from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

#from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced


# In[150]:


pip install -U imbalanced-learn


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Imported Libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
#from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


# In[152]:


# statisics summary of each feature
invoice_model.describe()


# In[153]:


# check the missing value
invoice_model.isnull().sum().max()


# In[154]:


invoice_model.info()


# In[155]:


#split the traning data and test data before we balance the dataset (0.2)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

#print('On Time', round(invoice_model['class'].value_counts()[0]/len(invoice_model) * 100,2), '% of the dataset')
#print(', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

X = invoice_model.drop('class', axis=1)
y = invoice_model['class']

#sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

#for train_index, test_index in sss.split(X, y):
 #   print("Train:", train_index, "Test:", test_index)
 #   original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
 #   original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.
original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the Distribution of the labels


# Turn into an array
original_Xtrain_1 = original_Xtrain.values
original_Xtest_1 = original_Xtest.values
original_ytrain_1 = original_ytrain.values
original_ytest_1 = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain_1, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest_1, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain_1))
print(test_counts_label/ len(original_ytest_1))


# In[156]:


original_Xtrain


# In[157]:


original_ytrain


# In[ ]:


# create the training data
original_Xtrain['class'] = original_ytrain


# In[159]:


# df is our training data
df = original_Xtrain.copy()
df


# In[160]:


# check the distribution of our training data
z = df.groupby("class").count()
z 


# In[161]:


# balance our training data
# Random Under-Sampling:
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.

# Lets shuffle the data before creating the subsamples

df = df.sample(frac=1)

zero_df = df.loc[df['class'] == 0][:13349]
one_df = df.loc[df['class'] == 1][:13349]
two_df = df.loc[df['class'] == 2][:13349]
three_df = df.loc[df['class'] == 3][:13349]
four_df = df.loc[df['class'] == 4]

normal_distributed_df = pd.concat([zero_df, one_df, two_df, three_df, four_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df.head()


# In[162]:


# check the distribution of each class after undersampling
new_df.groupby("class").count()


# In[163]:


# distribution of the classes after undersampling
# now the training data is balance
print('Distribution of the Classes in the subsample dataset')
print(new_df['class'].value_counts()/len(new_df))



sns.countplot('class', data=new_df, palette=colors)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


# In[164]:


# check our new traning data
new_df


# In[165]:


# Make sure we use the subsample in our correlation
# check the correlation for each feature

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

# Entire DataFrame
corr = invoice_model.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)


sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()


# In[ ]:


# use training data to train the model
X = new_df.drop('class', axis=1) # features
y = new_df['class'] # target


# In[167]:


# import library
from collections import Counter

from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

#from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

# Load Library
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
print(__doc__)

RANDOM_STATE = 42

# import library
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


# In[ ]:


##train and evaluate to find the best model


# In[169]:


##model 1: KNN

#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.neighbors import NearestNeighbors

##Create a pipeline

#pipeline4 = make_pipeline(NearMiss(version=2),
#                        KNeighborsClassifier(n_neighbors = 5))
#pipeline4.fit(X, y)

#cv_scores = cross_val_score(pipeline, X, y, cv=10)

#print each cv score (accuracy) and average them

#print(cv_scores)
#print('cv_scores mean:{}'.format(np.mean(cv_scores)))
#print(classification_report(original_ytest, pipeline4.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline4.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline4.predict(original_Xtest)))


# In[170]:


##model 2:  Naive Bayes

#from sklearn.naive_bayes import GaussianNB
#pipeline5 = make_pipeline(NearMiss(version=2),
#                       GaussianNB())
#pipeline5.fit(X, y)

#cv_scores = cross_val_score(pipeline, X, y, cv=10)
##print each cv score (accuracy) and average them

#print(cv_scores)
#print('cv_scores mean:{}'.format(np.mean(cv_scores)))
#print(classification_report(original_ytest, pipeline5.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline5.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline5.predict(original_Xtest)))


# In[171]:


##model 3: Logistic Regression

#pipeline3 = make_pipeline(NearMiss(version=2),
#                        LogisticRegression(random_state=RANDOM_STATE))
#pipeline3.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline3.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline3.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline3.predict(original_Xtest)))


# In[172]:


##model 4: SVM

#pipeline2 = make_pipeline(NearMiss(version=2),
#                        LinearSVC(random_state=RANDOM_STATE))
#pipeline2.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline2.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline2.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline2.predict(original_Xtest)))


# In[173]:


##model 5: Random Forest

#pipeline = make_pipeline(NearMiss(version=2),
#                        RandomForestClassifier())
#pipeline.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))


# In[ ]:


##model 6: GradientBoostingClassifier

#pipeline1 = make_pipeline(NearMiss(version=2),
#                        GradientBoostingClassifier())
#pipeline1.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline1.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline1.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline1.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline1.predict(original_Xtest))
#print(kappa)

##result:
##GradientBoostingClassifier is the best


# In[174]:


##model 7: Ada Boost

#pipeline6 = make_pipeline(NearMiss(version=2),
#                       AdaBoostClassifier())
#pipeline6.fit(X, y)

#print(classification_report(original_ytest, pipeline6.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline6.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline6.predict(original_Xtest)))


# In[175]:


##model 8: Decision Tree

#pipeline7 = make_pipeline(NearMiss(version=2),
#                       DecisionTreeClassifier())
#pipeline7.fit(X, y)

#print(classification_report(original_ytest, pipeline7.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline7.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline7.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score
#kappa = cohen_kappa_score(original_ytest, pipeline3.predict(original_Xtest))
#kappa


# In[180]:


##model 9: xgboost

#from sklearn import datasets
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.svm import LinearSVC

#from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
#from xgboost.sklearn import XGBClassifier
#import sklearn
#import numpy as np
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.metrics import precision_score,roc_auc_score
#min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))

##fit transform(partData)

#Xtrans = min_max_scaler.fit_transform(X)
#Xtesttrans_new = min_max_scaler.fit_transform(original_Xtest)

##xgboost model

#model = OneVsRestClassifier(XGBClassifier(),n_jobs=2)
#clf = model.fit(Xtrans, y)

##predict

#pre_Y = clf.predict(Xtesttrans_new)

##model accuracy
#print(accuracy_score(original_ytest, pre_Y))


# In[181]:


##xgboost classification report / confusion matrix / kappa
##xgboost has bad performance on predict class 2, class 3, class 4

#print(confusion_matrix(original_ytest, pre_Y))
#print(classification_report(original_ytest, pre_Y))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pre_Y)
#kappa


# In[ ]:


##GradientBoostingClassifier + OneVsOneClassifier

#from sklearn.multiclass import OneVsOneClassifier

#GradientBoostingClassifier
#pipeline9 = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier()))
#pipeline9.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline9.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline9.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline9.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline9.predict(original_Xtest))
#print(kappa)

##result:
##OneVsOneClassifier is better than OneVsRestClassifier


# In[ ]:


##GradientBoostingClassifier + OneVsRestClassifier

#from sklearn.multiclass import OneVsRestClassifier

#GradientBoostingClassifier
#pipeline10 = make_pipeline(NearMiss(version=2),
#                        OneVsRestClassifier(GradientBoostingClassifier()))
#pipeline10.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline10.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline10.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline10.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline10.predict(original_Xtest))
#print(kappa)

##result:
##OneVsOneClassifier is better than OneVsRestClassifier


# In[ ]:


#from sklearn.metrics import hamming_loss
#ham_distance = hamming_loss(original_ytest, pipeline9.predict(original_Xtest))
#ham_distance


# In[ ]:


##'n_estimators'=200

#from sklearn.multiclass import OneVsOneClassifier

# GradientBoostingClassifier
#pipeline = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=200)))
#pipeline.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))

#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline.predict(original_Xtest))
#kappa


# In[ ]:


##n_estimators=300

#pipeline = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=300)))
#pipeline.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))

#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline.predict(original_Xtest))
#print(kappa)


# In[ ]:


##'n_estimators'=400

#pipeline = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=400)))
#pipeline.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))

#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline.predict(original_Xtest))
#print(kappa)


# In[ ]:


##'n_estimators'=500

#pipeline = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=500)))
#pipeline.fit(X, y)

#Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))

#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline.predict(original_Xtest))
#print(kappa)


# In[ ]:


# 'n_estimators'=600

#pipeline = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=600)))
#pipeline.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))

#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline.predict(original_Xtest))
#print(kappa)


# In[ ]:


##'n_estimators'=700

#pipeline = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=700)))
#pipeline.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline.predict(original_Xtest))
#print(kappa)


# In[ ]:


##'n_estimators'=800

#pipeline = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=800)))
#pipeline.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))

#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline.predict(original_Xtest))
#print(kappa)


# In[ ]:


##'n_estimators'=900

#pipeline = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=900)))
#pipeline.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline.predict(original_Xtest))
#print(kappa)


# In[ ]:


##'n_estimators'=1000

#pipeline = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=1000)))
#pipeline.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline.predict(original_Xtest))
#print(kappa)


# In[ ]:


##Grid Search: 'n_estimators' from 300 to 900

#param_test1 = {'n_estimators':range(300,801,100)}
#gsearch1 = GridSearchCV(GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
#                                  min_samples_leaf=20,max_depth=10,max_features='sqrt', subsample=0.8,random_state=10), 
#                     param_grid = param_test1,iid=False,cv=5)
#gsearch1.fit(X,y)
#gsearch1.best_params_, gsearch1.best_score_

##results
##n_estimators = 300 is the best, score:0.6241366394486478


# In[ ]:


##Grid Search: 'n_estimators' from 300 to 400

#param_test1 = {'n_estimators':range(300,401,10)}
#gsearch1 = GridSearchCV(GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
#                                  min_samples_leaf=20,max_depth=10,max_features='sqrt', subsample=0.8,random_state=10),  
#                     param_grid = param_test1,iid=False,cv=5)
#gsearch1.fit(X,y)
#gsearch1.best_params_, gsearch1.best_score_

##results
##'n_estimators=300 is still the best, score:0.6241366394486478


# In[ ]:


##Grid Search: 'max_depths' from 3 to 14

#param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
#gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=300, min_samples_leaf=20, 
#      max_features='sqrt', subsample=0.8, random_state=10), 
#   param_grid = param_test2,iid=False, cv=5)
#gsearch2.fit(X,y)
#gsearch2.best_params_, gsearch2.best_score_

##results
##max_depths = 5 is the best
##'min_samples_split' = 300 is the best
##score:0.6357629785002622


# In[ ]:


##GradientBoostingClassifier
##'n_estimators'=300, 'max_depth'=5

#pipeline = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=300, max_depth=5)))
#pipeline.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline.predict(original_Xtest))
#print(kappa)


# In[ ]:


##Grid Search: 'min_samples_split' from 800 to 1900
##Grid Search: 'min_samples_leaf' from 60 to 101

#param_test3 = {'min_samples_split':range(800,1900,200), 'min_samples_leaf':range(60,101,10)}
#gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=300,max_depth=5,
#                                     max_features='sqrt', subsample=0.8, random_state=10), 
#                       param_grid = param_test3,iid=False, cv=5)
#gsearch3.fit(X,y)
#gsearch3.best_params_, gsearch3.best_score_

##results
##'min_samples_split' = 800 is the best
##'min_samples_leaf' = 100 is the best
##score: 0.6351936474642296


# In[ ]:


##GradientBoostingClassifier
##'n_estimators'=300,  'min_samples_split'=800, 'min_samples_leaf'=100, 'max_depth'=5                      

#pipeline = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=300,  min_samples_split=800,
#                                  min_samples_leaf=100, max_depth=5)))
#pipeline.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))

#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline.predict(original_Xtest))
#print(kappa)


# In[ ]:


##Grid Search: 'max_features' from 7 to 18

#param_test4 = {'max_features':range(7,19,1)}
#gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=300,max_depth=5, min_samples_leaf =800, 
#               min_samples_split =100, subsample=0.8, random_state=10), 
#                       param_grid = param_test4, iid=False, cv=5)
#gsearch4.fit(X,y)
#gsearch4.best_params_, gsearch4.best_score_

##result:
##({'max_features': 8}, 0.634010038205109)


# In[ ]:


##best model
##'n_estimators'=300,  'min_samples_split'=800, 'min_samples_leaf'=100, 'max_depth'=5, 'max_features'=8

#pipeline11 = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=300,  min_samples_split=800,
#                                  min_samples_leaf=100, max_depth=5, max_features=8)))
#pipeline11.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline11.predict(original_Xtest)))

#print(accuracy_score(original_ytest, pipeline11.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline11.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline11.predict(original_Xtest))
#print(kappa)


# In[ ]:


##'n_estimators'=600,  'min_samples_split'=800, 'min_samples_leaf'=100, 'max_depth'=5, 'max_features'=8
 
#pipeline11 = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=600,  min_samples_split=800,
#                                  min_samples_leaf=100, max_depth=5, max_features=8)))
#pipeline11.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline11.predict(original_Xtest)))
#print(accuracy_score(original_ytest, pipeline11.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline11.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline11.predict(original_Xtest))
#print(kappa)


# In[ ]:


##Grid Search: 'subsample':[0.5,0.6,0.7,0.8]

#param_test5 = {'subsample':[0.5,0.6,0.7,0.8]}
#gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=300,  min_samples_split=800,
#                                  min_samples_leaf=100, max_depth=5, max_features=8, random_state=10), 
#                       param_grid = param_test5,iid=False, cv=5)
#gsearch5.fit(X,y)
#gsearch5.best_params_, gsearch5.best_score_

##result:
##({'subsample': 0.7}, 0.6348041051764177)


# In[ ]:


##GradientBoostingClassifier
##'n_estimators'=300,  'min_samples_split'=800, 'min_samples_leaf'=100, 'max_depth'=5, 'max_features'=8, 'subsample'=0.7

#pipeline12 = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=300,  min_samples_split=800,
#                                  min_samples_leaf=100, max_depth=5, max_features=8, subsample=0.7)))
#pipeline12.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline12.predict(original_Xtest)))

#print(accuracy_score(original_ytest, pipeline12.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline12.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline12.predict(original_Xtest))
#print(kappa)


# In[ ]:


##GradientBoostingClassifier
##'learning_rate'=0.05, 'n_estimators'=300,  'min_samples_split'=800, 'min_samples_leaf'=100, 'max_depth'=5, 'max_features'=8, 'subsample'=0.7, 'random_state'=10

#gbm2 = OneVsOneClassifier(GradientBoostingClassifier(learning_rate=0.05, n_estimators=300,  min_samples_split=800,
#                                  min_samples_leaf=100, max_depth=5, max_features=8, subsample=0.7, random_state=10))
#gbm2.fit(X,y)
#y_pred = gbm2.predict(original_Xtest)

#print(classification_report(original_ytest, y_pred))
#print(accuracy_score(original_ytest, y_pred))
#print(confusion_matrix(original_ytest, y_pred))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, y_pred)
#print(kappa)


# In[ ]:


##GradientBoostingClassifier
##'learning_rate'=0.1, 'n_estimators'=300,  'min_samples_split'=800, 'min_samples_leaf'=100, 'max_depth'=5, 'max_features'=8, 'subsample'=0.7, 'random_state'=10

#gbm = OneVsOneClassifier(GradientBoostingClassifier(learning_rate=0.1, n_estimators=300,  min_samples_split=800,
#                                  min_samples_leaf=100, max_depth=5, max_features=8, subsample=0.7, random_state=10))
#gbm.fit(X,y)
#y_pred = gbm.predict(original_Xtest)

#print(classification_report(original_ytest, y_pred))
#print(accuracy_score(original_ytest, y_pred))
#print(confusion_matrix(original_ytest, y_pred))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, y_pred)
#print(kappa)


# In[ ]:


##GradientBoostingClassifier
##'learning_rate'=0.01, 'n_estimators'=300,  'min_samples_split'=800, 'min_samples_leaf'=100, 'max_depth'=5, 'max_features'=8, 'subsample'=0.7, 'random_state'=10

#gbm1 = OneVsOneClassifier(GradientBoostingClassifier(learning_rate=0.01, n_estimators=300,  min_samples_split=800,
#                                  min_samples_leaf=100, max_depth=5, max_features=8, subsample=0.7, random_state=10))
#gbm1.fit(X,y)
#y_pred = gbm1.predict(original_Xtest)

#print(classification_report(original_ytest, y_pred))
#print(accuracy_score(original_ytest, y_pred))
#print(confusion_matrix(original_ytest, y_pred))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, y_pred)
#print(kappa)


# In[ ]:


##GradientBoostingClassifier
##'learning_rate'=0.005, 'n_estimators'=300,  'min_samples_split'=800, 'min_samples_leaf'=100, 'max_depth'=5, 'max_features'=8, 'subsample'=0.7, 'random_state'=10

#gbm3 = OneVsOneClassifier(GradientBoostingClassifier(learning_rate=0.005, n_estimators=300,  min_samples_split=800,
#                                  min_samples_leaf=100, max_depth=5, max_features=8, subsample=0.7, random_state=10))
#gbm3.fit(X,y)
#y_pred = gbm3.predict(original_Xtest)

#print(classification_report(original_ytest, y_pred))
#print(accuracy_score(original_ytest, y_pred))
#print(confusion_matrix(original_ytest, y_pred))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, y_pred)
#print(kappa)


# In[ ]:


##GradientBoostingClassifier
##'learning_rate'=0.1, 'n_estimators'=300,

#pipeline = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(learning_rate=0.1,n_estimators=300)))
#pipeline.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))

#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline.predict(original_Xtest))
#print(kappa)


# In[ ]:


##GradientBoostingClassifier
##'learning_rate'=0.05, 'n_estimators'=300,

#pipeline = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(learning_rate=0.05,n_estimators=300)))
#pipeline.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline.predict(original_Xtest)))

#print(accuracy_score(original_ytest, pipeline.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline.predict(original_Xtest))
#print(kappa)


# In[ ]:


##GradientBoostingClassifier
##'learning_rate'=0.1, 'n_estimators'=300, 'min_samples_split'=800,'min_samples_leaf'=100, 'max_depth'=5, 'max_features'=8

#pipeline11 = make_pipeline(NearMiss(version=2),
#                        OneVsOneClassifier(GradientBoostingClassifier(learning_rate=0.1,n_estimators=300,  min_samples_split=800,
#                                  min_samples_leaf=100, max_depth=5, max_features=8)))
#pipeline11.fit(X, y)

##Classify and report the results

#print(classification_report(original_ytest, pipeline11.predict(original_Xtest)))

#print(accuracy_score(original_ytest, pipeline11.predict(original_Xtest)))
#print(confusion_matrix(original_ytest, pipeline11.predict(original_Xtest)))

#from sklearn.metrics import cohen_kappa_score

#kappa = cohen_kappa_score(original_ytest, pipeline11.predict(original_Xtest))
#print(kappa)


# In[207]:


##best model
##evaluate the performance of our selected model

pipeline_final = make_pipeline(NearMiss(version=2),
                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=300,  min_samples_split=800,
                                  min_samples_leaf=100, max_depth=5, max_features=8, random_state=42)))
pipeline_final.fit(X, y)
y_pred = pipeline_final.predict(original_Xtest)

##Classify and report the results

print(classification_report(original_ytest, y_pred))

print(accuracy_score(original_ytest, y_pred))
print(confusion_matrix(original_ytest, y_pred))

from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(original_ytest, y_pred)
print(kappa)


# In[219]:


## create the test data result as csv file

#result = original_Xtest.copy()
#result['class_pred'] = y_pred
#result


# In[220]:


## compare the predict test data result with the correct result

#result['class_actual'] = original_ytest
#result


# In[ ]:


## export test data 'result' dataframe as csv file

#result.to_csv('gbcmodel_result.csv')


# In[ ]:


##Model for prediction 

pipeline_model = make_pipeline(NearMiss(version=2),
                        OneVsOneClassifier(GradientBoostingClassifier(n_estimators=300,  min_samples_split=800,
                                  min_samples_leaf=100, max_depth=5, max_features=8, random_state=42)))
pipeline_model.fit(X, y)


# In[ ]:


##predict our invoices(for the real invoices we would like to predict)

y_pred_model = pipeline_model.predict(df) #df is the name of invoice dataframe which you want to predict by our model


# In[ ]:


## create the prediction result as csv file

model_result = df.copy() #df is the name of invoice dataframe which you want to predict by our model
model_result['class_pred'] = y_pred_model
model_result


# In[ ]:


## export 'model_result' dataframe as csv file (after export the csv file, go to 'file' button on the left hand site, and click the csv file to download)

model_result.to_csv('gbcmodel_result.csv')

