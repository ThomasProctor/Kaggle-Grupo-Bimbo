
from __future__ import division


from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV 







from sklearn.metrics import mean_squared_error
import logging
import numpy as np
import pandas as pd
import pickle
import joblib
from time import time



from sklearn.cross_validation import cross_val_score
#######################PARAMETERS#######################################
maxcats=5000
nfiles=10
########################################################################

logging.basicConfig(filename='Feature_Selection2.log',level=logging.INFO)
mytime = lambda: round(time())
t0=mytime()

with open('datapath.txt','r') as f:
    datapath=f.readlines()[0].rstrip()

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

splitfilenames=[splitpath+'%i_split_train.csv' % i for i in xrange(nfiles)]

frames=[pd.read_csv(fl,**kwargs) for fl in splitfilenames]

train=pd.concat(frames)

#train=pd.read_csv(splitpath+'0_split_train.csv',parse_dates=False,dtype=traindtypes,usecols=columns)

logging.info('%i sec: Data imported' % mytime()-t0)
# In[14]:

categoricals='Agencia_ID Canal_ID Ruta_SAK Cliente_ID Producto_ID'.split()




Clientes=pd.read_csv(datapath+'cliente_tabla.csv.zip',usecols=['Cliente_ID']).drop_duplicates().values.flatten()


# In[16]:

Productos=pd.read_csv(datapath+'producto_tabla.csv.zip',usecols=["Producto_ID"]).drop_duplicates().values.flatten()


# In[17]:

Agencias=pd.read_csv(datapath+'town_state.csv.zip',usecols=['Agencia_ID']).drop_duplicates().values.flatten()


# In[18]:

categories={'Agencia_ID':Agencias.tolist(),'Canal_ID':intcount['Canal_ID'].index.tolist(),'Ruta_SAK':intcount['Ruta_SAK'].index.tolist(),'Cliente_ID':Clientes.tolist(),'Producto_ID':Productos.tolist()}


for i in categories.keys():
	train.loc[:,i]=train[i].astype('category',categories=categories[i])

logging.info("%i sec: Categories categorized" % mytime()-t0)

categoricals='Agencia_ID Canal_ID Ruta_SAK Producto_ID Cliente_ID'.split()



def n_cutoff(s,counts,n):
    if s.cat.categories.shape[0]<=n:
        return None
    else:
        counts=counts.sort_values(ascending=False)
        print(s.name)
        drop_cats=set(counts.iloc[n:].index.tolist())
        print(len(drop_cats))
        drop_cats=list(drop_cats.intersection(set(s.cat.categories.tolist())))
        print(len(drop_cats))
        s.cat.remove_categories(drop_cats,inplace=True)

with open('Cliente_ID_Stats.pkl','r') as f:
    cuenta_Cliente=pickle.load(f)
    
for i in categoricals:
	if i in intcount.keys():
		n_cutoff(train[i], intcount[i], maxcats)

n_cutoff(train['Cliente_ID'],cuenta_Cliente,maxcats)

absent_products=[i for i in Productos if i not in intcount['Producto_ID'].index.tolist()]

train['Producto_ID'].cat.remove_categories(absent_products,inplace=True)

catsize={i:train[i].cat.categories.shape[0] for i in categoricals}

logging.info('%i sec: categories reduced:' % mytime()-t0)
logging.info(str(catsize))

def rmsle(estimator,X,y_true):
    y_pre=estimator.predict(X)
    return np.sqrt(mean_squared_error(np.log(y_pre+1),np.log(y_true+1)))
    
def rmse(estimator,X,y_true):
    y_pre=estimator.predict(X)
    return np.sqrt(mean_squared_error(y_pre,y_true))
    





# In[148]:

#trainsamp=train.sample(frac=0.2)


# In[86]:







# In[126]:

def standardize(s):
    if s.std()>0.0:
        return ((s-s.mean())/s.std())
    else:
        return s



train['Lunes']=((train['Semana']==7)).astype(np.int32).to_sparse(fill_value=int(0))
train['Miercoles-Jueves']=((train['Semana']==3)|(train['Semana']==9)).astype(np.int32).to_sparse(fill_value=int(0))




X=pd.get_dummies(train.drop(['Demanda_uni_equil','Semana'],axis=1),sparse=True,columns=categoricals,prefix_sep=":")
X=X.apply(standardize)
logging.info('%i sec: Data dummified' % mytime()-t0)
Xcols=X.columns

y=np.log(train['Demanda_uni_equil']+1.0)


# Lasso:

lasso=LassoCV()

lasso.fit(X,y)

logging.info('%i sec: Regression fit' % mytime()-t0)

coefs=pd.Series({Xcols[i]:lasso.coef_[i] for i in xrange((Xcols).shape[0])})


coefs.to_csv('0_split_default_ridge_coefs.csv')


joblib.dump(lasso,'%i_lassoCV_model.pkl' % nfiles)

cvscore_lasso=cross_val_score(lasso,X,y,scoring=rmse)



logging.info('Lasso error: %s' % str(cvscore_lasso))


# Random Forest

samp=X.join(y).sample(frac=0.05)

rfr=RandomForestRegressor(n_jobs=-1)

params1={'max_features':np.linspace(0.1,1.0,num=3),
       'max_depth':np.linspace(10,100,num=3,dtype=int).tolist()+[None],
       }
params2={'min_samples_split':np.linspace(2,10,num=4),
        'min_weight_fraction_leaf':np.linspace(0.0,0.5,num=4),
}

n_ests={'n_estimators':np.logspace(2,4,num=3,dtype=int)}

paramlist=[param1,param2,n_ests]

#        'max_leaf_nodes':np.linspace(10,1000,num=4,dtype=int).tolist()+[None]

gskwarg={"n_jobs":2,"pre_dispatch":2}
i=0
for param in paramlist:
	grid=GridSearchCV(rfr,param,**gskwarg)
	grid.fit(samp.drop('Demanda_uni_equil',axis=1),samp['Demanda_uni_equil'])
	rfr=grid.best_estimator_
	i+=1
	logging.info("%i sec: Grid iteration %i" % (mytime()-t0,i))

rfr.fit(X,y)

logging.info("%i sec: Random forest final fit" % mytime()-t0)

joblib.dump(rfr,'%i_randfor_model.pkl' % nfiles)

cvscore_rfr=cross_val_score(rfr,X,y,scoring=rmse)

pd.DataFrame(











