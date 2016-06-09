
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from math import ceil
import pickle
with open('datapath.txt') as f:
    datapath=f.readlines()[0].rstrip()




# In[4]:

dtypes={'Cliente_ID':np.uint32}

clientes_ID=pd.Index(pd.read_csv(datapath+'cliente_tabla.csv.zip',parse_dates=False).drop_duplicates(subset='Cliente_ID')['Cliente_ID'])


# In[5]:

def addcounts(cnt,cnk,index):
    newcnt=df_counts(cnk)
    return (cnt + newcnt.reindex(index=index, fill_value=0.0)).astype(np.uint32)


# In[6]:

def df_counts(df):
    return df.groupby('Cliente_ID').size().astype(np.uint32)







# In[8]:

probcol=['Cliente_ID']


# In[9]:

chunksz=10000
trainit=pd.read_csv(datapath+'train.csv',chunksize=chunksz,iterator=True,usecols=probcol,dtype=dtypes,parse_dates=False)


# In[ ]:

#%%time
ichunk=0
for chunk in trainit:
    if ichunk==0:
        probcount=df_counts(chunk).reindex(index=clientes_ID, fill_value=0.0)
        #sample=chunk.sample(frac=samplerate)
    else:
        probcount=addcounts(probcount,chunk,clientes_ID)
    if ichunk % 100 == 0:
        print(ichunk)
        print probcount.shape
    ichunk+=1

with open('Cliente_ID_Stats.pkl','w') as f:
    pickle.dump(probcount, f)
# In[ ]:



