```python
import numpy as np
import os
from time import time
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sklearn
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold,ShuffleSplit,GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from joblib import dump, load
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_squared_log_error,r2_score
from sklearn_pandas import DataFrameMapper
from pandas.api.types import is_string_dtype, is_numeric_dtype



from scipy.stats import skew,randint
from scipy.special import boxcox1p

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
                  log_y=True,
                  id_col= None,
                  drop_target=True,
                  convert_to_cat_cols=None,
                  remove_skewness=False,scale_mapper=None,
                  bin_columns_dict=None,
                  new_features_func=None):
    
    if drop_target:
        df,y = extract_and_drop_target_column(df_train,'SalePrice',inplace=True)
    if log_y:
        y = np.log1p(y)
    else:
        y = None
        
    combined = pd.concat((df, df_test)).reset_index(drop=True)
    
    
    if id_col is not None:
        combined.drop(id_col, axis=1,inplace=True)
        if df_test is not None:
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
    
    
    if remove_skewness:
        skewed_cols = combined[num_cols].apply(lambda x: skew(x)).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew' :skewed_cols})
        skewness_log = skewness[skewness > 4.0]
        skewness_other = skewness[skewness <= 4.0]
        skewed_features_log = skewness_log.index
        skewed_features_other = skewness_other.index
        lambda_ = 0.0
        for feature in skewed_features_log:
            combined[feature] = boxcox1p(combined[feature],lambda_)
        lambda_ = 0.15
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
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    return df
def add_new_features2(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    return df
def add_new_features3(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    df['TotalLivArea'] = df['GrLivArea'] + df['GarageArea'] + df['LotArea']
    return df

def add_new_features4(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    df['TotalLivArea'] = df['GrLivArea'] + df['GarageArea'] + df['LotArea']
    
    df["GrLivArea-2"] = df["GrLivArea"] ** 2
    df["GrLivArea-3"] = df["GrLivArea"] ** 3
    df["GrLivArea-Sq"] = np.sqrt(df["GrLivArea"])
    df["GarageArea-2"] = df["GarageArea"] ** 2
    df["GarageArea-3"] = df["GarageArea"] ** 3
    df["GarageArea-Sq"] = np.sqrt(df["GarageArea"])
    return df

    
```


```python
df_raw = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)
df = df_raw.copy()
stratify_col = df['OverallQual'].copy()
combined,df,y,cat_cols,num_cols,test_id,n_train,n_test = preprocess_df(
                                       df_train=df,df_test=df_test,
                                       drop_target=True,
                                       new_features_func=add_new_features4,
                                       id_col='Id',
                                       log_y=True,
                                       convert_to_cat_cols=['GarageCars','CentralAir','MoSold','YrSold','YearBuilt','GarageYrBlt','YearRemodAdd'],
                                       remove_skewness=True,
                                       scale_mapper=RobustScaler(),
                                       bin_columns_dict={'OverallQual':(1,11,10),'OverallCond':(1,11,10)} 
                                       )

```

    0.0
    


```python
df = combined[:n_train]
df_test = combined[n_train:]

X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.10,
                                  stratify=stratify_col,shuffle = True,random_state=20)

stratify_X_train = stratify_col[:X_train.shape[0]].copy()
X_train.shape,X_test.shape,y_train.shape,y_test.shape, stratify_X_train.shape
```




    ((1314, 679), (146, 679), (1314,), (146,), (1314,))




```python
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.10,
                                  stratify=stratify_X_train,shuffle = True,random_state=20)
X_train.shape,X_valid.shape,y_train.shape,y_valid.shape
```




    ((1182, 679), (132, 679), (1182,), (132,))




```python
rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model, X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.0028047344988199473, MSE Validation set = 0.01411351102318513, score Training Set = 0.982475518284114, score on Validation Set = 0.9002889431518937
    OOB Score = 0.8696681470748157
    


```python
gb_model = GradientBoostingRegressor(n_estimators=1500,random_state=10).fit(X_train,y_train)
print_mse(gb_model, X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.00011330964997227497, MSE Validation set = 0.014309458799005205, score Training Set = 0.9992920210843458, score on Validation Set = 0.8989045845906568
    


```python
elasticnet_model = ElasticNet(alpha=0.01,l1_ratio=.9,random_state=100).fit(X_train,y_train)
print_mse(elasticnet_model, X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.021835061589687035, MSE Validation set = 0.013969604890325386, score Training Set = 0.8635706382352015, score on Validation Set = 0.9013056308188246
    


```python
RandomForestRegressor()
```




    RandomForestRegressor()




```python
GradientBoostingRegressor()
```




    GradientBoostingRegressor()




```python
params = {'n_estimators':[300,500,800,1100,1500,1800],
              'max_features': [0.5,0.7,0.9,'auto'],
              'min_samples_split': [2,3,10],
              'min_samples_leaf': [1,3,10]}

start = time()
gridSearch_rf = GridSearchCV(RandomForestRegressor(warm_start=True,n_jobs=8),param_grid=params,n_jobs=8)        
gridSearch_rf.fit(X_train,y_train)
print('training took {} minutes'.format((time() - start)/60.))

print_mse(gridSearch_rf, X_train,X_valid,y_train,y_valid)
```

    training took 180.7719754735629 minutes
    MSE Training set = 0.0027635057829138695, MSE Validation set = 0.013763836095813053, score Training Set = 0.9827331226592765, score on Validation Set = 0.9027593742518712
    


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
report(gridSearch_rf.cv_results_)
```

    Model with rank: 1
    Mean validation score: 0.869 (std: 0.021)
    Parameters: {'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1100}
    
    Model with rank: 2
    Mean validation score: 0.868 (std: 0.021)
    Parameters: {'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 1100}
    
    Model with rank: 3
    Mean validation score: 0.868 (std: 0.021)
    Parameters: {'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
    
    


```python
dump(gridSearch_rf,'gridSearch_rf_iowa.pkl')
```




    ['gridSearch_rf_iowa.pkl']




```python
gridSearch_rf = load('gridSearch_rf_iowa.pkl')
```


```python
kfold = KFold(n_splits=5, shuffle=True, random_state=0)
```


```python
params = {'n_estimators':[300,500,800,1100,1500,1800],
              "max_features": randint(80,680),
              "min_samples_split": randint(2, 11),
              "min_samples_leaf": randint(1, 11),
              "subsample":[0.6,0.7,0.75,0.8,0.9]
         }

randomSearch_gb = RandomizedSearchCV(GradientBoostingRegressor(warm_start=True),
                                     param_distributions=params,n_iter=20,
                                     cv=kfold,n_jobs=6)        
randomSearch_gb.fit(X_train,y_train)
print_mse(randomSearch_gb, X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.002688116048416019, MSE Validation set = 0.012840422121941909, score Training Set = 0.9832041711753942, score on Validation Set = 0.9092832351885116
    


```python
dump(randomSearch_gb,'randomSearch_gb_iowa.pkl')
```




    ['randomSearch_gb_iowa.pkl']




```python
randomSearch_gb = load('randomSearch_gb_iowa.pkl')
```


```python
report(randomSearch_gb.cv_results_)
```

    Model with rank: 1
    Mean validation score: 0.897 (std: 0.018)
    Parameters: {'max_features': 445, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'subsample': 0.9}
    
    Model with rank: 2
    Mean validation score: 0.895 (std: 0.017)
    Parameters: {'max_features': 550, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 300, 'subsample': 0.7}
    
    Model with rank: 3
    Mean validation score: 0.893 (std: 0.019)
    Parameters: {'max_features': 666, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 1100, 'subsample': 0.75}
    
    


```python
params = {'n_estimators':[300,500,800,1100,1500,1800],
              "max_features": randint(80,680),
              "min_samples_split": randint(2, 11),
              "min_samples_leaf": randint(1, 11)
         }

randomSearch_rf = RandomizedSearchCV(RandomForestRegressor(warm_start=True),
                                     param_distributions=params,cv=kfold,
                                     n_jobs=6, n_iter=20)        
randomSearch_rf.fit(X_train,y_train)
print_mse(randomSearch_rf, X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.0029771946304933444, MSE Validation set = 0.014051322431847132, score Training Set = 0.9813979566020719, score on Validation Set = 0.9007283015904862
    


```python
dump(randomSearch_rf,'randomSearch_rf_iowa.pkl')
```




    ['randomSearch_rf_iowa.pkl']




```python
randomSearch_rf = load('randomSearch_rf_iowa.pkl')
```


```python
report(randomSearch_rf.cv_results_)
```

    Model with rank: 1
    Mean validation score: 0.872 (std: 0.020)
    Parameters: {'max_features': 559, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 800}
    
    Model with rank: 2
    Mean validation score: 0.872 (std: 0.021)
    Parameters: {'max_features': 466, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 300}
    
    Model with rank: 3
    Mean validation score: 0.872 (std: 0.024)
    Parameters: {'max_features': 238, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 300}
    
    


```python
ElasticNet()
```




    ElasticNet()




```python
params = {'alpha':[0.001,0.01,0.1,1.],
          'l1_ratio': [0.4,0.5,0.6,0.7,0.8,0.9],
          'max_iter':[1000,2000,5000,10000],
          'selection':['cyclic','random']
         }

randomSearch_elastic = RandomizedSearchCV(ElasticNet(warm_start=True),param_distributions=params,
                                          cv=kfold,n_jobs=6, n_iter=20)        
randomSearch_elastic.fit(X_train,y_train)
print_mse(randomSearch_elastic, X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.012659795133610223, MSE Validation set = 0.009900352486589855, score Training Set = 0.9208993405831607, score on Validation Set = 0.9300546399839869
    


```python
dump(randomSearch_elastic,'randomSearch_elastic_iowa.pkl')
```




    ['randomSearch_elastic_iowa.pkl']




```python
randomSearch_elastic = load('randomSearch_elastic_iowa.pkl')
```


```python
report(randomSearch_elastic.cv_results_)
```

    Model with rank: 1
    Mean validation score: 0.888 (std: 0.027)
    Parameters: {'selection': 'cyclic', 'max_iter': 5000, 'l1_ratio': 0.8, 'alpha': 0.001}
    
    Model with rank: 2
    Mean validation score: 0.888 (std: 0.027)
    Parameters: {'selection': 'cyclic', 'max_iter': 1000, 'l1_ratio': 0.9, 'alpha': 0.001}
    
    Model with rank: 3
    Mean validation score: 0.888 (std: 0.027)
    Parameters: {'selection': 'random', 'max_iter': 5000, 'l1_ratio': 0.9, 'alpha': 0.001}
    
    


```python
randomSearch_elastic1 = ElasticNet(alpha=0.001,
                                   selection='cyclic',
                                   max_iter=5000,
                                   l1_ratio=0.8
                                   ).fit(X_train,y_train)
print_mse(randomSearch_elastic1,X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.012659795133610223, MSE Validation set = 0.009900352486589855, score Training Set = 0.9208993405831607, score on Validation Set = 0.9300546399839869
    


```python
randomSearch_elastic2 = ElasticNet(alpha=0.001,
                                   selection='cyclic',
                                   max_iter=1000,
                                   l1_ratio=0.9
                                   ).fit(X_train,y_train)
print_mse(randomSearch_elastic2,X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.013059788602674633, MSE Validation set = 0.009933692858180533, score Training Set = 0.9184001100007143, score on Validation Set = 0.9298190923812995
    


```python
randomSearch_elastic3 = ElasticNet(alpha=0.001,
                                   selection='random',
                                   max_iter=5000,
                                   l1_ratio=0.9
                                   ).fit(X_train,y_train)
print_mse(randomSearch_elastic3,X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.013060491220521254, MSE Validation set = 0.00994412648425177, score Training Set = 0.9183957199190109, score on Validation Set = 0.9297453794773585
    


```python
print_mse(randomSearch_elastic1,X_train,X_test,y_train,y_test)
```

    MSE Training set = 0.012659795133610223, MSE Validation set = 0.012284430576120175, score Training Set = 0.9208993405831607, score on Validation Set = 0.9258003527311824
    


```python
rfe_model = RFE(randomSearch_elastic1).fit(X_train,y_train)
print_mse(rfe_model, X_train,X_test,y_train,y_test)
```

    MSE Training set = 0.012659791759553347, MSE Validation set = 0.012284440103317432, score Training Set = 0.9208993616648702, score on Validation Set = 0.9258002951856029
    


```python
print('GridSearch RF')
print_mse(gridSearch_rf, X_train,X_test,y_train,y_test)
print('RandomSearch RF:')
print_mse(randomSearch_rf, X_train,X_test,y_train,y_test)
print('RandomSearch GB:')
print_mse(randomSearch_gb, X_train,X_test,y_train,y_test)
print('RandomSearch Elastic:')
print_mse(randomSearch_elastic, X_train,X_test,y_train,y_test)
print('RFE Elastic:')
print_mse(rfe_model, X_train,X_test,y_train,y_test)
```

    GridSearch RF
    MSE Training set = 0.0027635057829138664, MSE Validation set = 0.019900423978304223, score Training Set = 0.9827331226592765, score on Validation Set = 0.8797987069453199
    RandomSearch RF:
    MSE Training set = 0.0029771946304933444, MSE Validation set = 0.019851023984718993, score Training Set = 0.9813979566020719, score on Validation Set = 0.8800970896889394
    RandomSearch GB:
    MSE Training set = 0.002688116048416019, MSE Validation set = 0.01504534434221217, score Training Set = 0.9832041711753942, score on Validation Set = 0.9091240545247457
    RandomSearch Elastic:
    MSE Training set = 0.012659795133610223, MSE Validation set = 0.012284430576120175, score Training Set = 0.9208993405831607, score on Validation Set = 0.9258003527311824
    RFE Elastic:
    MSE Training set = 0.012659791759553347, MSE Validation set = 0.012284440103317432, score Training Set = 0.9208993616648702, score on Validation Set = 0.9258002951856029
    


```python

```
