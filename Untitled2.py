
# coding: utf-8

# In[284]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


# In[285]:


import seaborn as sns


# In[436]:


df = pd.read_excel('C:/Users/ABHI/Downloads/Train_dataset.xlsx')


# In[437]:


df.head(2)


# In[438]:


df.columns


# In[439]:


df['Index'].unique()


# In[440]:


df['Index'].nunique()


# In[441]:


def index(col):
    if col=="NYSE":
        return 1
    if col == "BSE":
        return 2
    if col == "S&P 500":
        return 3
    if col == "NSE":
        return 4
    if col=="JSE":
        return 5
    


# In[442]:


df['Index'] = df['Index'].apply(index)


# In[443]:


df.head()


# In[444]:


df.info()


# In[445]:


sns.distplot(df['Stock Price'])


# In[446]:


def genindex(col):
    a = col[0]
    b = col[1]
    if pd.isnull(b):
        if a==1:
            return 12765.84
        if a ==2:
            return 38182.08
        if a==3:
            return 3351.28
        if a==4:
            return 11270.15
        if a==5:
            return 55722 
    else:
        return b


# In[447]:


df['General Index'] = df[['Index','General Index']].apply(genindex,axis = 1)


# In[448]:


df['Industry'].nunique()


# In[449]:


df['Dollar Exchange Rate'].nunique()


# In[450]:


def doler(col):
    a = col[0]
    b = col[1]
    if pd.isnull(b):
        if a==1 or a==3:
            return 1
        if a ==2 or a==4:
            return 74.9
        if a ==5:
            return 17.7
    else:
        return b


# In[451]:


df['Dollar Exchange Rate'] = df[['Index','Dollar Exchange Rate']].apply(doler,axis=1)


# In[452]:


df.head(2)


# In[453]:


df['Industry'].unique()


# In[454]:


def industry(col):
    if col=="Real Estate":
        return 1
    if col=="Information Tech":
        return 2
    if col=="Materials":
        return 3
    if col=="Healthcare":
        return 4
    if col=="Energy":
        return 5


# In[455]:


sns.heatmap(df.isnull(),cmap="viridis")


# In[456]:


df['Industry'] = df['Industry'].apply(industry)


# In[457]:


def covid(col):
    a =col[0]
    b = col[1]
    if pd.isnull(b):
        if a==1:
            return -0.43
        if a==2:
            return 0.23
        if a==3:
            return 0.03
        if a==4:
            return 0.78
        if a==5:
            return 0.11
    else:
        return b
        


# In[458]:


df['Covid Impact (Beta)'] = df[['Industry','Covid Impact (Beta)']].apply(covid,axis=1)


# In[459]:


df.replace(r'^\s*$', np.nan, regex=True)


# In[460]:


df.head()


# In[461]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[462]:


df.drop(columns=['Stock Index'],inplace=True)


# In[463]:


imp = IterativeImputer(max_iter=100,random_state=0)
imp.fit(df)
IterativeImputer(random_state=0)
df[:] = imp.transform(df)


# In[464]:


X = df.drop(columns = ['Stock Price'])
y = df['Stock Price']


# In[465]:


X.head(2)


# In[42]:


sns.pairplot(X)


# In[466]:


X.describe()


# In[467]:


X.corr()


# In[468]:


test = pd.read_excel('C:/Users/ABHI/Downloads/Test_dataset.xlsx')


# In[469]:


sns.heatmap(test.isnull(),cmap='viridis')


# In[470]:


test.head()


# In[471]:


test['Index'] = test['Index'].apply(index)


# In[472]:


test.head()


# In[473]:


test['General Index'] = test[['Index','General Index']].apply(genindex,axis = 1)


# In[474]:


test['Dollar Exchange Rate'] = test[['Index','Dollar Exchange Rate']].apply(doler,axis=1)


# In[475]:


test['Industry'] = test['Industry'].apply(industry)


# In[476]:


test['Covid Impact (Beta)'] = test[['Industry','Covid Impact (Beta)']].apply(covid,axis=1)


# In[477]:


test.drop(columns=['Stock Index'],inplace=True)


# In[478]:


test.replace(r'^\s*$', np.nan, regex=True)


# In[479]:


imp1 = IterativeImputer(max_iter=100,random_state=0)
imp1.fit(test)
IterativeImputer(random_state=0)
test[:] = imp1.transform(test)


# In[480]:


test.head(2)


# In[481]:


sns.heatmap(test.isnull())


# In[482]:


test.head(2)


# In[483]:


x = test


# In[484]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()


# In[485]:


X = sc_X.fit_transform(X)


# In[486]:


from sklearn.model_selection import train_test_split


# In[545]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[546]:


from sklearn.ensemble import RandomForestRegressor


# In[557]:


rfg = RandomForestRegressor()


# In[558]:


rfg.fit(X_train,y_train)


# In[561]:


param_grid = {'n_estimators':[170,350,200,400,500,550,600,680,760,840,920]}


# In[562]:


from sklearn.model_selection import RandomizedSearchCV


# In[563]:


random_cv = RandomizedSearchCV(rfg,param_distributions=param_grid,n_iter=10,verbose=3,random_state=0,return_train_score=True)


# In[564]:


random_cv.fit(X_train,y_train)


# In[584]:


random_cv.best_estimator_


# In[585]:


random_cv.best_score_


# In[540]:


random_cv.best_params_


# In[586]:


rfg = RandomForestRegressor(n_estimators=760)


# In[587]:


rfg.fit(X_train,y_train)


# In[588]:


rfg_pred = rfg.predict(X_test)


# In[589]:


np.mean(np.abs(rfg_pred/y_test -1)*100)


# In[590]:


from sklearn import metrics


# In[591]:


metrics.mean_absolute_error(y_test,rfg_pred)


# In[592]:


plt.figure(figsize = (10,8))
plt.plot(y_test,rfg_pred,color = 'blue',ls='dashed',marker='o',markerfacecolor = 'red',markersize=10)


# In[593]:


sc_x = StandardScaler()


# In[594]:


x = sc_x.fit_transform(x)


# In[595]:


rfg.fit(X,y)


# In[596]:


pred = rfg.predict(x)


# In[597]:


pred


# In[598]:


rfg_pred


# In[599]:


y_test.head()


# In[400]:


######################################################################################### Task 2


# In[600]:


test1 = pd.read_excel('C:/Users/ABHI/Downloads/Test_dataset.xlsx',sheet_name="Put-Call_TS")


# In[601]:


test1.head()


# In[602]:


test1.drop(columns=['Stock Index'],inplace=True)


# In[603]:


test1 = test1.iloc[1:,:]


# In[604]:


test1.replace(r'^\s*$',np.nan,regex=True)


# In[605]:


imp2 = IterativeImputer(max_iter=50,random_state=0)
imp2.fit(test1)
IterativeImputer(random_state=0)
test1[:] = imp2.transform(test1)


# In[606]:


d = test1.iloc[:,:5]


# In[607]:


y1 = test1.iloc[:,5]


# In[608]:


d_train,d_test,y1_train,y1_test = train_test_split(d,y1,test_size=0.3,random_state=101)


# In[609]:


rfg1 = RandomForestRegressor(n_estimators=88)


# In[610]:


rfg1.fit(d_train,y1_train)


# In[611]:


dum_pred = rfg1.predict(d_test)


# In[612]:


np.mean(np.abs(dum_pred/y1_test -1)*100)


# In[613]:


d1 = test1.iloc[:,1:]


# In[614]:


rfg1.fit(d,y1)


# In[615]:


real_pred = rfg1.predict(d1)


# In[616]:


test['Put-Call Ratio'] = real_pred


# In[617]:


test = sc_x.fit_transform(test)


# In[618]:


final_pred = rfg.predict(test)


# In[619]:


final_pred


# In[620]:


plt.figure(figsize = (10,8))
plt.plot(pred,final_pred,color = 'blue',ls='dashed',marker='o',markerfacecolor = 'red',markersize=10)


# In[622]:


surya1 = np.loadtxt('C:/Users/ABHI/Downloads/y_predct_test.csv')


# In[623]:


surya1


# In[624]:


surya2 = np.loadtxt('C:/Users/ABHI/Downloads/final_predict.csv')


# In[625]:


surya2


# In[627]:


pred


# In[628]:


final_pred


# In[706]:


file = pd.DataFrame(pred,columns=['Stock Price(10th August)'])


# In[707]:


test_n = pd.read_excel('C:/Users/ABHI/Downloads/Test_dataset.xlsx')


# In[719]:


test_n = pd.DataFrame(test_n.iloc[:,0])


# In[720]:


test_n.head()


# In[710]:


test_n['Stock Price(10th August)'] = file


# In[711]:


test_n.to_csv('Part-01Solution Data.csv')


# In[718]:


file1 = pd.DataFrame(final_pred,columns=['Stock Price(16th August)'])


# In[721]:


test_n['Stock Price(16th August)'] = file1


# In[722]:


test_n.head()


# In[723]:


test_n.to_csv('Part-02Solution Data.csv')

