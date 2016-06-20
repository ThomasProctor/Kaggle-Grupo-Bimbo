
# coding: utf-8

# In[22]:

import numpy as np
import pandas as pd
import pickle
from time import time
import matplotlib.pyplot as plt


# In[4]:

with open('datapath.txt','r') as f:
    datapath=f.readlines()[0].rstrip()


# In[5]:

# In[2]:

with open('catcount.pkl','r') as f:
    floatstats, intcount = pickle.load(f)


# In[6]:

splitpath=datapath+'SplitFiles/'

# In[3]:

intcount={i.index.name:i for i in intcount}




traindtypes=pd.read_csv(splitpath+'0_split_train.csv',nrows=10,parse_dates=False).dtypes.to_dict()
traindtypes['Semana']=np.uint8



# In[7]:

columns=['Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK',
       'Cliente_ID', 'Producto_ID', 'Demanda_uni_equil']

kwargs={'parse_dates':False,'dtype':traindtypes,'usecols':columns}


# In[8]:

train=pd.read_csv(splitpath+'0_split_train.csv', **kwargs)['Cliente_ID']


# In[9]:

train.shape


# In[10]:

Clientes=pd.read_csv(datapath+'cliente_tabla.csv.zip',usecols=['Cliente_ID']).drop_duplicates().values.flatten()


# In[12]:

Clientes.shape


# In[13]:

train=train.astype('category', categories=Clientes.tolist())


# In[18]:

def n_cutoff(s,counts,n):
    if s.cat.categories.shape[0]<=n:
        return None
    else:
        counts=counts.sort_values(ascending=False)
        #print(s.name)
        drop_cats=set(counts.iloc[n:].index.tolist())
        #print(len(drop_cats))
        drop_cats=list(drop_cats.intersection(set(s.cat.categories.tolist())))
        #print(len(drop_cats))
        s.cat.remove_categories(drop_cats,inplace=True)


# In[17]:

with open('Cliente_ID_Stats.pkl','r') as f:
    cuenta_Cliente=pickle.load(f)
    


# In[39]:

def time_dummies(n):
    X=train.copy()
    n_cutoff(X,cuenta_Cliente,n)
    print(X.cat.categories.shape)
    t0=time()
    pd.get_dummies(X,sparse=True)
    return time()-t0


# In[ ]:

timevncats=pd.Series({i:time_dummies(i) for i in xrange(50,200,50)},name='time')
timevncats.index.name='categories'
timevncats.to_csv('time_to_dummify.csv')





# In[31]:

from statsmodels.api import OLS


# In[49]:

X=pd.DataFrame([timevncats.index.to_series(),timevncats.index.to_series()**2],index='x x**2'.split()).T






import statsmodels.api as sm


# In[65]:

ols=OLS(timevncats,sm.add_constant(X))


# In[66]:

ols=ols.fit()





nclients=Clientes.shape[0]

predtime=(ols.predict([1,nclients,nclients**2])/60/60)[0]

print('Full data set should take %i hours' % int(predtime))






