```python
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from time import time
import sklearn
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold,ShuffleSplit,StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_selection import RFE,SelectPercentile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from joblib import dump, load
from sklearn.metrics import confusion_matrix,roc_auc_score, make_scorer
from sklearn_pandas import DataFrameMapper
from pandas.api.types import is_string_dtype, is_numeric_dtype
from scipy import stats
from scipy.stats import skew,randint
from scipy.special import boxcox1p
from scipy.stats import randint as sp_randint
%matplotlib inline
```


```python
def print_feature_importances(model,X):
    important_features = pd.Series(data=rf_model.feature_importances_,index=X.columns)
    important_features.sort_values(ascending=False,inplace=True)
    print(important_features.head(50))
    
def get_cat_columns_by_type(df):
    out = []
    for colname,col_values in df.items():
        if is_string_dtype(col_values):
            out.append((colname,'string') )
        elif not is_numeric_dtype(col_values):
            out.append((colname,'categorical') )
    return out       

def get_numeric_columns(df):
    out = []
    for colname,col_values in df.items():
        if is_numeric_dtype(col_values):
            out.append(colname)
    return out       
    
def get_missing_values_percentage(df):
    missing_values_counts_list = df.isnull().sum()
    total_values = np.product(df.shape)
    total_missing = missing_values_counts_list.sum()
    # percent of data that is missing
    return (total_missing/total_values) * 100

def get_missing_columns(df1,df2):
    missing1 = []
    missing2 = []
    for colname in df1.columns:
        if colname not in df2.columns:
            missing2.append(colname)
    for colname in df2.columns:
        if colname not in df1.columns:
            missing1.append(colname)        
    return (missing1,missing2)


def convert_to_str_type(df_in,columns,inplace=False):
    if(inplace):
        df = df_in
    else:
        df = df_in.copy()
        
    for col in columns:
        df[col] = df[col].astype(str)
    return df

    
def handle_missing_values(df_in,cat_cols=[], num_cols=[],na_dict=None,add_nan_col=True,inplace=False):
    if(inplace):
        df = df_in
    else:
        df = df_in.copy()
 
    if na_dict is None:
        na_dict = {}

    for colname, col_values in df.items():   
        if colname not in num_cols:
            continue
        if pd.isnull(col_values).sum():
            df[colname+'_na'] = pd.isnull(col_values)
            filler = na_dict[colname] if colname in na_dict else col_values.median()
            df[colname] = col_values.fillna(filler)
            na_dict[colname] = filler
    for colname in cat_cols:
        if colname not in df.columns:
            continue
        df[colname].fillna(df[colname].mode()[0], inplace=True)
        lbl = LabelEncoder() 
        lbl.fit(list(df[colname].values)) 
        df[colname] = lbl.transform(list(df[colname].values))
    
    return (df,na_dict)



def scale_num_cols(df_in, mapper, inplace=False):
    if(inplace):
        df = df_in
    else:
        df = df_in.copy()
        
    if mapper is None:
        map_f = [([c],StandardScaler()) for c in df.columns if is_numeric_dtype(df[c])]
        mapper = DataFrameMapper(map_f).fit(df)
    df[mapper.transformed_names_] = mapper.transform(df)
    return (df,mapper)



def extract_and_drop_target_column(df_in, y_name, inplace=False):
    if(inplace):
        df = df_in
    else:
        df = df_in.copy()
    if not is_numeric_dtype(df[y_name]):
        df[y_name] = df[y_name].cat.codes
        y = df[y_name].values
    else:
        y = df[y_name]
    df.drop([y_name], axis=1, inplace=True)
    return (df,y)

def print_mse(m,X_train, X_valid, y_train, y_valid):
    res = [mean_squared_error(y_train,m.predict(X_train)),
                mean_squared_error(y_valid,m.predict(X_valid)),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    print('MSE Training set = {}, MSE Validation set = {}, score Training Set = {}, score on Validation Set = {}'.format(res[0],res[1],res[2], res[3]))
    if hasattr(m, 'oob_score_'):
          print('OOB Score = {}'.format(m.oob_score_))      

def get_iqr_min_max(df,cols):
    out = {}
    for colname, col_values in df.items():
        if colname not in cols:
            continue
        quartile75, quartile25 = np.percentile(col_values, [75 ,25])
        ## Inter Quartile Range ##
        IQR = quartile75 - quartile25
        min_value = quartile25 - (IQR*1.5)
        max_value = quartile75 + (IQR*1.5)
        out[colname] = (min_value,max_value)
    return out


def bin_numerical_columns(df_in,cols,inplace=False):
    if(inplace):
        df = df_in
    else:
        df = df_in.copy()
        
    for col in cols.keys():
        bins = cols[col]
        buckets_ = np.linspace(bins[0],bins[1],bins[2])
        df[col] = pd.cut(df[col],buckets_,include_lowest=True)
    return df
```


```python
def preprocess_df(df_train,df_test=None,
                  log_y=False,
                  id_col= None,test_id=None,
                  target_col=None,
                  convert_to_cat_cols=None,
                  remove_skewness=False,
                  skew_threshold=0.75,
                  boxcox_lambda=0.15,
                  scale_mapper=None,
                  bin_columns_dict=None,
                  new_features_func=None):
    
    if target_col is not None:
        df,y = extract_and_drop_target_column(df_train,target_col,inplace=True)
        print(y.head())
        if log_y:
            y = np.log1p(y)
            
    else:
        y = None
        
        
    combined = pd.concat((df, df_test)).reset_index(drop=True)
    
    
    if id_col is not None:
        combined.drop(id_col, axis=1,inplace=True)
        if test_id is not None:
            test_id = df_test[id_col].copy()
        else: test_id = None
   
    if new_features_func is not None:
        combined = new_features_func(combined)
    
    
    if convert_to_cat_cols is not None:
        combined = convert_to_str_type(combined,convert_to_cat_cols,inplace=True)
    
        
    if bin_columns_dict is not None:
        combined = bin_numerical_columns(combined,bin_columns_dict,inplace=True)
    
    
    cat_cols = get_cat_columns_by_type(combined)
    cat_cols = [cat_cols[i][0] for i in range(len(cat_cols))]
    num_cols = [col for col in combined.columns if col not in cat_cols]
    
    combined = pd.get_dummies(combined,columns=cat_cols, dummy_na=True)
    
    n_train = df.shape[0]
    n_test = df_test.shape[0]
      
    
    combined,d = handle_missing_values(combined,cat_cols=cat_cols,
                                       num_cols=num_cols,inplace=True)
    
    print(d)
    if remove_skewness:
        skewed_cols = combined[num_cols].apply(lambda x: skew(x)).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew' :skewed_cols})
        skewness_log = skewness[abs(skewness) > skew_threshold]
        skewness_other = skewness[abs(skewness) <= skew_threshold]
        skewed_features_log = skewness_log.index
        skewed_features_other = skewness_other.index
        lambda_ = 0.0
        for feature in skewed_features_log:
            combined[feature] = boxcox1p(combined[feature],lambda_)
        lambda_ = boxcox_lambda
        for feature in skewed_features_other:
            combined[feature] = boxcox1p(combined[feature],lambda_)
    
    if scale_mapper is not None:
        map_f = [([c],scale_mapper) for c in num_cols]
        mapper = DataFrameMapper(map_f).fit(combined)
    else:
        mapper = None
        
    combined,_ = scale_num_cols(combined,mapper,inplace=True) 
    
    print(get_missing_values_percentage(combined))
    
    return combined,df,y,cat_cols,num_cols,test_id,n_train,n_test

```


```python
def add_new_features1(df):
    df['DepsIncomeComined'] = df['NumberOfDependents'] * df['MonthlyIncome']
    df['Times90DaysLateDebtRatio'] = df['NumberOfTimes90DaysLate'] * df['DebtRatio']
    df['Times90DaysLateRevolving'] = df['NumberOfTimes90DaysLate'] * df['RevolvingUtilizationOfUnsecuredLines']
    return df
def add_new_features2(df):
    df['DepsIncomeComined'] = df['NumberOfDependents'] * df['MonthlyIncome']
    df['Times90DaysLateDebtRatio'] = df['NumberOfTimes90DaysLate'] * df['DebtRatio']
    df['Times90DaysLateRevolving'] = df['NumberOfTimes90DaysLate'] * df['RevolvingUtilizationOfUnsecuredLines']
    df['RevolvingUtilizationOfUnsecuredLines-2'] = df['RevolvingUtilizationOfUnsecuredLines'] ** 2
    df['RevolvingUtilizationOfUnsecuredLines-3'] = df['RevolvingUtilizationOfUnsecuredLines'] ** 3
    df['RevolvingUtilizationOfUnsecuredLines-sqrt'] = np.sqrt(df['RevolvingUtilizationOfUnsecuredLines'])
    
    return df

```


```python

df_raw = pd.read_csv('cs-train.csv', low_memory=False)
df_test = pd.read_csv('cs-test.csv', low_memory=False)
columns = ['Id', 'SeriousDlqin2yrs','RevolvingUtilizationOfUnsecuredLines', 'age',
                 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                 'NumberOfDependents']
df_raw.columns= columns
df_test.columns = columns
df_test.drop(['SeriousDlqin2yrs'], axis=1, inplace=True)
df = df_raw.copy()
combined,df,y,cat_cols,num_cols,test_id,n_train,n_test = preprocess_df(
                                       df_train=df,df_test=df_test,
                                       target_col='SeriousDlqin2yrs',
                                       id_col='Id',test_id='Id',
                                       convert_to_cat_cols=[
                                       'NumberOfTime30-59DaysPastDueNotWorse',
                                       'NumberOfTime60-89DaysPastDueNotWorse'
                                       ],
                                       new_features_func=add_new_features2,
                                       remove_skewness=True,
                                       skew_threshold=0.75,
                                       boxcox_lambda=0.2,
                                       scale_mapper=RobustScaler()
                                       )

df = combined[:n_train]
df_test = combined[n_train:]
stratify_col = y

X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.10,
                                  stratify=y,shuffle = True,random_state=20)

stratify_X_train = stratify_col[:X_train.shape[0]].copy()
X_train.shape,X_test.shape,y_train.shape,y_test.shape, stratify_X_train.shape
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.2,
                                  stratify=stratify_X_train,shuffle = True,random_state=20)
X_train.shape,X_valid.shape,y_train.shape,y_valid.shape
```

    0    1
    1    0
    2    0
    3    0
    4    0
    Name: SeriousDlqin2yrs, dtype: int64
    {'MonthlyIncome': 5400.0, 'NumberOfDependents': 0.0, 'DepsIncomeComined': 0.0}
    0.0
    




    ((108000, 49), (27000, 49), (108000,), (27000,))




```python
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
```


```python
params = {'n_estimators':[200,300,500,800,1000,1500],
              "max_features": randint(10,50),
              "min_samples_split": randint(2, 11),
              "min_samples_leaf": randint(1, 11)
              
         }
start = time()
randomSearch_rf = RandomizedSearchCV(RandomForestClassifier(warm_start=True,n_jobs=6),scoring='roc_auc',param_distributions=params,n_iter=20,n_jobs=6)        
randomSearch_rf.fit(X_train,y_train)
print('training took {} mins'.format((time() - start)/60.))
randomSearch_rf_auc = roc_auc_score(y_valid,randomSearch_rf.predict_proba(X_valid)[:, 1])
print("AUC for Randomized Search Random Forest: {:.6f}".format(randomSearch_rf_auc))
```

    training took 27.8490231672923 mins
    AUC for Randomized Search Random Forest: 0.853726
    


```python
dump(randomSearch_rf,'randomSearch_rf_credit.pkl')
```




    ['randomSearch_rf_credit.pkl']




```python
randomSearch_rf = load('randomSearch_rf_credit.pkl')
```


```python
report(randomSearch_rf.cv_results_)
```

    Model with rank: 1
    Mean validation score: 0.857 (std: 0.003)
    Parameters: {'max_features': 27, 'min_samples_leaf': 9, 'min_samples_split': 5, 'n_estimators': 500}
    
    Model with rank: 2
    Mean validation score: 0.856 (std: 0.004)
    Parameters: {'max_features': 46, 'min_samples_leaf': 10, 'min_samples_split': 7, 'n_estimators': 500}
    
    Model with rank: 3
    Mean validation score: 0.856 (std: 0.003)
    Parameters: {'max_features': 13, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 500}
    
    


```python
randomSearch_rf_auc = roc_auc_score(y_valid,randomSearch_rf.predict_proba(X_valid)[:, 1])
print("AUC for Randomized Search Random Forest: {:.6f}".format(randomSearch_rf_auc))
```

    AUC for Randomized Search Random Forest: 0.853726
    


```python
rf_model_rank1 = RandomForestClassifier(n_estimators=1500,
                                        max_features=15,
                                        min_samples_leaf=9,min_samples_split=8,
                                        n_jobs=6).fit(X_train,y_train)

dump(rf_model_rank1,'rf_model_credit_rank1.pkl')
```




    ['rf_model_credit_rank1.pkl']




```python
rf_model_rank2 = RandomForestClassifier(n_estimators=1000,
                                        max_features=18,
                                        min_samples_leaf=8,min_samples_split=8,
                                        n_jobs=6).fit(X_train,y_train)
dump(rf_model_rank2,'rf_model_credit_rank2.pkl')
```




    ['rf_model_credit_rank2.pkl']




```python
rf_model_rank3 = RandomForestClassifier(n_estimators=200,
                                        max_features=27,
                                        min_samples_leaf=10,min_samples_split=6,
                                        n_jobs=6).fit(X_train,y_train)
dump(rf_model_rank3,'rf_model_credit_rank3.pkl')
```




    ['rf_model_credit_rank3.pkl']




```python
rf_model_rank1 = load('rf_model_credit_rank1.pkl')
rf_model_rank2 = load('rf_model_credit_rank2.pkl')
rf_model_rank3 = load('rf_model_credit_rank3.pkl')
```


```python
rf_model_rank1_auc = roc_auc_score(y_valid,rf_model_rank1.predict_proba(X_valid)[:, 1])
print("AUC for Randomized Search RF Rank 1: {:.6f}".format(rf_model_rank1_auc))
```

    AUC for Randomized Search RF Rank 1: 0.854837
    


```python
rf_model_rank2_auc = roc_auc_score(y_valid,rf_model_rank2.predict_proba(X_valid)[:, 1])
print("AUC for Randomized Search RF Rank 2: {:.6f}".format(rf_model_rank2_auc))
```

    AUC for Randomized Search RF Rank 2: 0.854069
    


```python
rf_model_rank3_auc = roc_auc_score(y_valid,rf_model_rank3.predict_proba(X_valid)[:, 1])
print("AUC for Randomized Search RF Rank 3: {:.6f}".format(rf_model_rank3_auc))
```

    AUC for Randomized Search RF Rank 3: 0.853182
    


```python
params = {'n_estimators':[200,300,500,800,1000,1500],
              "max_features": randint(10,50),
              "min_samples_split": randint(2, 11),
              "min_samples_leaf": randint(1, 11),
              "subsample":[0.6,0.7,0.75,0.8,0.9]
         }
start = time()
randomSearch_gb = RandomizedSearchCV(GradientBoostingClassifier(warm_start=True),scoring='roc_auc',param_distributions=params,n_iter=20,n_jobs=6)        
randomSearch_gb.fit(X_train,y_train)
print('training took {} mins'.format((time() - start)/60.))
```

    training took 5.39767011006673 mins
    


```python
dump(randomSearch_gb,'randomSearch_gb_credit.pkl')
```




    ['randomSearch_gb_credit.pkl']




```python
randomSearch_gb = load('randomSearch_gb_credit.pkl')
```


```python
randomSearch_gb_auc = roc_auc_score(y_valid,randomSearch_gb.predict_proba(X_valid)[:, 1])
print("AUC for Randomized Search Gradient Boost: {:.6f}".format(randomSearch_gb_auc))
```

    AUC for Randomized Search Gradient Boost: 0.861305
    


```python
report(randomSearch_gb.cv_results_)
```

    Model with rank: 1
    Mean validation score: 0.865 (std: 0.003)
    Parameters: {'max_features': 26, 'min_samples_leaf': 9, 'min_samples_split': 5, 'n_estimators': 500, 'subsample': 0.6}
    
    Model with rank: 2
    Mean validation score: 0.864 (std: 0.003)
    Parameters: {'max_features': 17, 'min_samples_leaf': 7, 'min_samples_split': 2, 'n_estimators': 200, 'subsample': 0.8}
    
    Model with rank: 3
    Mean validation score: 0.864 (std: 0.003)
    Parameters: {'max_features': 35, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 500, 'subsample': 0.75}
    
    


```python
gb_model_rank1 = GradientBoostingClassifier(max_features=34,min_samples_leaf=4,
                                           min_samples_split=3, n_estimators=300,
                                           subsample=0.75,
                                           warm_start=True).fit(X_train,y_train)

gb_model_rank2 = GradientBoostingClassifier(max_features=31,min_samples_leaf=8,
                                           min_samples_split=2, n_estimators=200,
                                           subsample=0.6,
                                           warm_start=True).fit(X_train,y_train)

gb_model_rank3 = GradientBoostingClassifier(max_features=23,min_samples_leaf=3,
                                           min_samples_split=6, n_estimators=200,
                                           subsample=0.8,
                                           warm_start=True).fit(X_train,y_train)

```


```python
dump(gb_model_rank1,'gb_model_credit_rank1.pkl')
dump(gb_model_rank2,'gb_model_credit_rank2.pkl')
dump(gb_model_rank3,'gb_model_credit_rank3.pkl')
```




    ['gb_model_credit_rank3.pkl']




```python
gb_model_rank1 = load('gb_model_credit_rank1.pkl')
gb_model_rank2 = load('gb_model_credit_rank2.pkl')
gb_model_rank3 = load('gb_model_credit_rank3.pkl')

```


```python
gb_model_rank1_auc = roc_auc_score(y_valid,gb_model_rank1.predict_proba(X_valid)[:, 1])
print("AUC for Randomized Search Gradient Boost 1: {:.6f}".format(gb_model_rank1_auc))
```

    AUC for Randomized Search Gradient Boost 1: 0.860985
    


```python
gb_model_rank2_auc = roc_auc_score(y_valid,gb_model_rank2.predict_proba(X_valid)[:, 1])
print("AUC for Randomized Search Gradient Boost 2: {:.6f}".format(gb_model_rank2_auc))

```

    AUC for Randomized Search Gradient Boost 2: 0.860445
    


```python
gb_model_rank3_auc = roc_auc_score(y_valid,gb_model_rank3.predict_proba(X_valid)[:, 1])
print("AUC for Randomized Search Gradient Boost 3: {:.6f}".format(gb_model_rank3_auc))

```

    AUC for Randomized Search Gradient Boost 3: 0.861308
    


```python
best_gb_auc = roc_auc_score(y_test,gb_model_rank3.predict_proba(X_test)[:, 1])
print("AUC for Random Forest: {:.6f}".format(best_gb_auc))
```

    AUC for Random Forest: 0.866756
    


```python
rfe_model = RFE(gb_model_rank3).fit(X_train,y_train)
dump(rfe_model,'rfe_model_credit.pkl')
```




    ['rfe_model_credit.pkl']




```python
rfe_model = load('rfe_model_credit.pkl')
```


```python
rfe_model_auc = roc_auc_score(y_valid,rfe_model.predict_proba(X_valid)[:, 1])
print("AUC for GB RFE: {:.6f}".format(rfe_model_auc))
```

    AUC for GB RFE: 0.860592
    


```python
preds_rfe_model = rfe_model.predict(X_valid)
confusion = confusion_matrix(y_valid, preds_rfe_model)
print("Confusion matrix:\n{}".format(confusion))
```

    Confusion matrix:
    [[24883   256]
     [ 1500   361]]
    


```python
preds_gb_rank1 = gb_model_rank1.predict(X_valid)
confusion = confusion_matrix(y_valid, preds_gb_rank1)
print("Confusion matrix:\n{}".format(confusion))
```

    Confusion matrix:
    [[24871   268]
     [ 1507   354]]
    


```python
preds_gb_rank2 = gb_model_rank2.predict(X_valid)
confusion = confusion_matrix(y_valid, preds_gb_rank2)
print("Confusion matrix:\n{}".format(confusion))
```

    Confusion matrix:
    [[24879   260]
     [ 1513   348]]
    


```python
preds_gb_rank3 = gb_model_rank3.predict(X_valid)
confusion = confusion_matrix(y_valid, preds_gb_rank3)
print("Confusion matrix:\n{}".format(confusion))
```

    Confusion matrix:
    [[24880   259]
     [ 1508   353]]
    


```python
knn_model = KNeighborsClassifier(n_neighbors=150,n_jobs=-1).fit(X_train,y_train)
dump(knn_model,'knn_model.pkl')
```




    ['knn_model.pkl']




```python
knn_model = load('knn_model.pkl')
```


```python
knn_model_auc = roc_auc_score(y_valid,knn_model.predict_proba(X_valid)[:, 1])
print("AUC for KNN: {:.6f}".format(knn_model_auc))
```

    AUC for KNN: 0.843305
    


```python
params = {'n_neighbors':[10,20,30,50,80,100,125,150,170],
              "weights": ['distance','uniform'],
              "p": [1,2]             
         }
start = time()
randomSearch_knn = RandomizedSearchCV(KNeighborsClassifier(n_jobs=-1),scoring='roc_auc',param_distributions=params,n_iter=20,n_jobs=-1)        
randomSearch_knn.fit(X_train,y_train)
print('training took {} mins'.format((time() - start)/60.))

```

    training took 2.69721782604853 mins
    


```python
report(randomSearch_knn.cv_results_)
```

    Model with rank: 1
    Mean validation score: 0.850 (std: 0.005)
    Parameters: {'weights': 'distance', 'p': 2, 'n_neighbors': 170}
    
    Model with rank: 2
    Mean validation score: 0.849 (std: 0.005)
    Parameters: {'weights': 'distance', 'p': 1, 'n_neighbors': 170}
    
    Model with rank: 3
    Mean validation score: 0.848 (std: 0.004)
    Parameters: {'weights': 'distance', 'p': 2, 'n_neighbors': 125}
    
    


```python
dump(randomSearch_knn,'randomSearch_knn.pkl')
```




    ['randomSearch_knn.pkl']




```python
randomSearch_knn = load('randomSearch_knn.pkl')
```


```python
randomSearch_knn_auc = roc_auc_score(y_valid,randomSearch_knn.predict_proba(X_valid)[:, 1])
print("AUC for Randomized Search KNN: {:.6f}".format(randomSearch_knn_auc))
```

    AUC for Randomized Search KNN: 0.842257
    


```python
knn_model_rank1 = KNeighborsClassifier(n_neighbors=50,n_jobs=-1,weights='distance',p=2).fit(X_train,y_train)
dump(knn_model_rank1,'knn_model_rank1.pkl')
```




    ['knn_model_rank1.pkl']




```python
knn_model_rank2 = KNeighborsClassifier(n_neighbors=50,n_jobs=-1,weights='uniform',p=2).fit(X_train,y_train)
dump(knn_model_rank2,'knn_model_rank2.pkl')
```




    ['knn_model_rank2.pkl']




```python
knn_model_rank3 = KNeighborsClassifier(n_neighbors=150,n_jobs=-1,weights='distance',p=2).fit(X_train,y_train)
dump(knn_model_rank3,'knn_model_rank3.pkl')
```




    ['knn_model_rank3.pkl']




```python
knn_model_rank1 = load('knn_model_rank1.pkl')
knn_model_rank2 = load('knn_model_rank2.pkl')
knn_model_rank3 =load('knn_model_rank3.pkl')
```


```python
knn_model_auc_rank1 = roc_auc_score(y_valid,knn_model_rank1.predict_proba(X_valid)[:, 1])
print("AUC for KNN 1: {:.6f}".format(knn_model_auc_rank1))

```

    AUC for KNN 1: 0.828132
    


```python
knn_model_auc_rank2 = roc_auc_score(y_valid,knn_model_rank2.predict_proba(X_valid)[:, 1])
print("AUC for KNN 2: {:.6f}".format(knn_model_auc_rank2))
```

    AUC for KNN 2: 0.830514
    


```python
knn_model_auc_rank3 = roc_auc_score(y_valid,knn_model_rank3.predict_proba(X_valid)[:, 1])
print("AUC for KNN 3: {:.6f}".format(knn_model_auc_rank3))
```

    AUC for KNN 3: 0.841437
    


```python
nb_model = GaussianNB().fit(X_train,y_train)
dump(nb_model,'nb_model.pkl')
```




    ['nb_model.pkl']




```python
nb_model = load('nb_model.pkl')
```


```python
nb_model_auc = roc_auc_score(y_valid,nb_model.predict_proba(X_valid)[:, 1])
print("AUC for NB on Valid: {:.6f}".format(nb_model_auc))
```

    AUC for NB on Valid: 0.844154
    


```python
nb_model_auc = roc_auc_score(y_test,nb_model.predict_proba(X_test)[:, 1])
print("AUC for NB on Test : {:.6f}".format(nb_model_auc))
```

    AUC for NB on Test : 0.843932
    


```python

```


```python

```


```python

```
