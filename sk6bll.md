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
from sklearn.feature_selection import RFE

from joblib import dump, load
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn_pandas import DataFrameMapper
from pandas.api.types import is_string_dtype, is_numeric_dtype
from scipy import stats
from scipy.stats import skew
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
        ## IQR ##
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
                  id_col= None,
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
            print('222')
    else:
        y = None
        print('333')
        
    combined = pd.concat((df, df_test)).reset_index(drop=True)
    
    
    if df_test is not None and id_col is not None:
        test_id = df_test['Id'].copy()
        combined.drop('Id', axis=1,inplace=True)
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
def add_new_features3(df):
    return df

def add_new_features4(df):
    return df

    
```


```python

df_raw = pd.read_csv('cs-train.csv', low_memory=False)
df_test = pd.read_csv('cs-test.csv', low_memory=False)
```


```python
df_raw.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150000 entries, 0 to 149999
    Data columns (total 12 columns):
     #   Column                                Non-Null Count   Dtype  
    ---  ------                                --------------   -----  
     0   Unnamed: 0                            150000 non-null  int64  
     1   SeriousDlqin2yrs                      150000 non-null  int64  
     2   RevolvingUtilizationOfUnsecuredLines  150000 non-null  float64
     3   age                                   150000 non-null  int64  
     4   NumberOfTime30-59DaysPastDueNotWorse  150000 non-null  int64  
     5   DebtRatio                             150000 non-null  float64
     6   MonthlyIncome                         120269 non-null  float64
     7   NumberOfOpenCreditLinesAndLoans       150000 non-null  int64  
     8   NumberOfTimes90DaysLate               150000 non-null  int64  
     9   NumberRealEstateLoansOrLines          150000 non-null  int64  
     10  NumberOfTime60-89DaysPastDueNotWorse  150000 non-null  int64  
     11  NumberOfDependents                    146076 non-null  float64
    dtypes: float64(4), int64(8)
    memory usage: 13.7 MB
    


```python
df_raw.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>SeriousDlqin2yrs</th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>1.202690e+05</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>146076.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>75000.500000</td>
      <td>0.066840</td>
      <td>6.048438</td>
      <td>52.295207</td>
      <td>0.421033</td>
      <td>353.005076</td>
      <td>6.670221e+03</td>
      <td>8.452760</td>
      <td>0.265973</td>
      <td>1.018240</td>
      <td>0.240387</td>
      <td>0.757222</td>
    </tr>
    <tr>
      <th>std</th>
      <td>43301.414527</td>
      <td>0.249746</td>
      <td>249.755371</td>
      <td>14.771866</td>
      <td>4.192781</td>
      <td>2037.818523</td>
      <td>1.438467e+04</td>
      <td>5.145951</td>
      <td>4.169304</td>
      <td>1.129771</td>
      <td>4.155179</td>
      <td>1.115086</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>37500.750000</td>
      <td>0.000000</td>
      <td>0.029867</td>
      <td>41.000000</td>
      <td>0.000000</td>
      <td>0.175074</td>
      <td>3.400000e+03</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>75000.500000</td>
      <td>0.000000</td>
      <td>0.154181</td>
      <td>52.000000</td>
      <td>0.000000</td>
      <td>0.366508</td>
      <td>5.400000e+03</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>112500.250000</td>
      <td>0.000000</td>
      <td>0.559046</td>
      <td>63.000000</td>
      <td>0.000000</td>
      <td>0.868254</td>
      <td>8.249000e+03</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>150000.000000</td>
      <td>1.000000</td>
      <td>50708.000000</td>
      <td>109.000000</td>
      <td>98.000000</td>
      <td>329664.000000</td>
      <td>3.008750e+06</td>
      <td>58.000000</td>
      <td>98.000000</td>
      <td>54.000000</td>
      <td>98.000000</td>
      <td>20.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_raw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>SeriousDlqin2yrs</th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.766127</td>
      <td>45</td>
      <td>2</td>
      <td>0.802982</td>
      <td>9120.0</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0.957151</td>
      <td>40</td>
      <td>0</td>
      <td>0.121876</td>
      <td>2600.0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>0.658180</td>
      <td>38</td>
      <td>1</td>
      <td>0.085113</td>
      <td>3042.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0.233810</td>
      <td>30</td>
      <td>0</td>
      <td>0.036050</td>
      <td>3300.0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>0.907239</td>
      <td>49</td>
      <td>1</td>
      <td>0.024926</td>
      <td>63588.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
columns = ['Id', 'SeriousDlqin2yrs','RevolvingUtilizationOfUnsecuredLines', 'age',
                 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                 'NumberOfDependents']
df_raw.columns= columns
df_test.columns = columns
df_test.drop(['SeriousDlqin2yrs'], axis=1, inplace=True)
df = df_raw.copy()
```


```python
#Find most important features relative to target
corr = df.corr()
corr.sort_values(['SeriousDlqin2yrs'], ascending = False, inplace = True)
print(corr.SeriousDlqin2yrs)

```

    SeriousDlqin2yrs                        1.000000
    NumberOfTime30-59DaysPastDueNotWorse    0.125587
    NumberOfTimes90DaysLate                 0.117175
    NumberOfTime60-89DaysPastDueNotWorse    0.102261
    NumberOfDependents                      0.046048
    Id                                      0.002801
    RevolvingUtilizationOfUnsecuredLines   -0.001802
    NumberRealEstateLoansOrLines           -0.007038
    DebtRatio                              -0.007602
    MonthlyIncome                          -0.019746
    NumberOfOpenCreditLinesAndLoans        -0.029669
    age                                    -0.115386
    Name: SeriousDlqin2yrs, dtype: float64
    


```python

combined,df,y,cat_cols,num_cols,test_id,n_train,n_test = preprocess_df(
                                       df_train=df,df_test=df_test,
                                       target_col='SeriousDlqin2yrs',
                                       id_col='Id',
                                       remove_skewness=True,
                                       skew_threshold=0.75,
                                       boxcox_lambda=0.2,
                                       scale_mapper=RobustScaler()
                                       )


```

    0    1
    1    0
    2    0
    3    0
    4    0
    Name: SeriousDlqin2yrs, dtype: int64
    {'MonthlyIncome': 5400.0, 'NumberOfDependents': 0.0}
    0.0
    


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150000 entries, 0 to 149999
    Data columns (total 11 columns):
     #   Column                                Non-Null Count   Dtype  
    ---  ------                                --------------   -----  
     0   Id                                    150000 non-null  int64  
     1   RevolvingUtilizationOfUnsecuredLines  150000 non-null  float64
     2   age                                   150000 non-null  int64  
     3   NumberOfTime30-59DaysPastDueNotWorse  150000 non-null  int64  
     4   DebtRatio                             150000 non-null  float64
     5   MonthlyIncome                         120269 non-null  float64
     6   NumberOfOpenCreditLinesAndLoans       150000 non-null  int64  
     7   NumberOfTimes90DaysLate               150000 non-null  int64  
     8   NumberRealEstateLoansOrLines          150000 non-null  int64  
     9   NumberOfTime60-89DaysPastDueNotWorse  150000 non-null  int64  
     10  NumberOfDependents                    146076 non-null  float64
    dtypes: float64(4), int64(7)
    memory usage: 12.6 MB
    


```python
df_raw['SeriousDlqin2yrs'].head()
```




    0    1
    1    0
    2    0
    3    0
    4    0
    Name: SeriousDlqin2yrs, dtype: int64




```python
y.head()
```




    0    1
    1    0
    2    0
    3    0
    4    0
    Name: SeriousDlqin2yrs, dtype: int64




```python
iqr_ranges = get_iqr_min_max(combined,num_cols)
iqr_ranges
```




    {'DebtRatio': (-0.6898509855776909, 1.8761315545181199),
     'MonthlyIncome': (4.768564869602322, 5.522452016090394),
     'NumberOfDependents': (-1.5, 2.5),
     'NumberOfOpenCreditLinesAndLoans': (1.300166838962975, 3.3058475995996432),
     'NumberOfTime30-59DaysPastDueNotWorse': (0.0, 0.0),
     'NumberOfTime60-89DaysPastDueNotWorse': (0.0, 0.0),
     'NumberOfTimes90DaysLate': (0.0, 0.0),
     'NumberRealEstateLoansOrLines': (-2.1583421225545587, 3.5972368709242644),
     'RevolvingUtilizationOfUnsecuredLines': (-0.9014455427324771,
      1.6425972222125693),
     'age': (2.9691248157200487, 3.8137150538445965)}




```python
combined.shape,df.shape,df_test.shape
```




    ((251503, 12), (150000, 11), (101503, 11))




```python
df = combined[:n_train]
df_test = combined[n_train:]
stratify_col = y

X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.10,
                                  stratify=y,shuffle = True,random_state=20)

stratify_X_train = stratify_col[:X_train.shape[0]].copy()
X_train.shape,X_test.shape,y_train.shape,y_test.shape, stratify_X_train.shape
```




    ((135000, 12), (15000, 12), (135000,), (15000,), (135000,))




```python
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.2,
                                  stratify=stratify_X_train,shuffle = True,random_state=20)
X_train.shape,X_valid.shape,y_train.shape,y_valid.shape
```




    ((108000, 12), (27000, 12), (108000,), (27000,))




```python
rf_model = RandomForestClassifier(n_estimators=300,random_state=10, n_jobs=-1).fit(X_train,y_train)
rf_auc = roc_auc_score(y_valid,rf_model.predict_proba(X_valid)[:, 1])
print("AUC for Random Forest: {:.6f}".format(rf_auc))
```

    AUC for Random Forest: 0.842681
    


```python
preds_rf = rf_model.predict(X_valid)
confusion = confusion_matrix(y_valid, preds_rf)
print("Confusion matrix:\n{}".format(confusion))
```

    Confusion matrix:
    [[24836   303]
     [ 1524   337]]
    


```python
gb_model = GradientBoostingClassifier(n_estimators=300,random_state=10).fit(X_train,y_train)
gb_auc = roc_auc_score(y_valid,gb_model.predict_proba(X_valid)[:, 1])
print("AUC for Gradient Boost: {:.6f}".format(gb_auc))
```

    AUC for Gradient Boost: 0.862963
    


```python
preds_gb = gb_model.predict(X_valid)
confusion = confusion_matrix(y_valid, preds_gb)
print("Confusion matrix:\n{}".format(confusion))
```

    Confusion matrix:
    [[24874   265]
     [ 1508   353]]
    


```python
print_feature_importances(rf_model,X_train)
```

    RevolvingUtilizationOfUnsecuredLines    0.191734
    DebtRatio                               0.174215
    MonthlyIncome                           0.140418
    age                                     0.129085
    NumberOfOpenCreditLinesAndLoans         0.090661
    NumberOfTimes90DaysLate                 0.088518
    NumberOfTime30-59DaysPastDueNotWorse    0.053856
    NumberOfTime60-89DaysPastDueNotWorse    0.048158
    NumberOfDependents                      0.041042
    NumberRealEstateLoansOrLines            0.034271
    MonthlyIncome_na                        0.005344
    NumberOfDependents_na                   0.002700
    dtype: float64
    


```python
print_feature_importances(gb_model,X_train)
```

    RevolvingUtilizationOfUnsecuredLines    0.191734
    DebtRatio                               0.174215
    MonthlyIncome                           0.140418
    age                                     0.129085
    NumberOfOpenCreditLinesAndLoans         0.090661
    NumberOfTimes90DaysLate                 0.088518
    NumberOfTime30-59DaysPastDueNotWorse    0.053856
    NumberOfTime60-89DaysPastDueNotWorse    0.048158
    NumberOfDependents                      0.041042
    NumberRealEstateLoansOrLines            0.034271
    MonthlyIncome_na                        0.005344
    NumberOfDependents_na                   0.002700
    dtype: float64
    


```python
df_raw['NumberOfTime30-59DaysPastDueNotWorse'].value_counts()
```




    0     126018
    1      16033
    2       4598
    3       1754
    4        747
    5        342
    98       264
    6        140
    7         54
    8         25
    9         12
    96         5
    10         4
    12         2
    11         1
    13         1
    Name: NumberOfTime30-59DaysPastDueNotWorse, dtype: int64




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
                                       id_col='Id',
                                       convert_to_cat_cols=[
                                       'NumberOfTime30-59DaysPastDueNotWorse',
                                       'NumberOfTime60-89DaysPastDueNotWorse'
                                       ],
                                       new_features_func=add_new_features1,
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
    




    ((108000, 46), (27000, 46), (108000,), (27000,))




```python
rf_model = RandomForestClassifier(n_estimators=300,random_state=10, n_jobs=-1).fit(X_train,y_train)
rf_auc = roc_auc_score(y_valid,rf_model.predict_proba(X_valid)[:, 1])
print("AUC for Random Forest: {:.6f}".format(rf_auc))
preds_rf = rf_model.predict(X_valid)
confusion = confusion_matrix(y_valid, preds_rf)
print("Confusion matrix:\n{}".format(confusion))
```

    AUC for Random Forest: 0.839244
    Confusion matrix:
    [[24844   295]
     [ 1521   340]]
    


```python
gb_model = GradientBoostingClassifier(n_estimators=300,random_state=10).fit(X_train,y_train)
gb_auc = roc_auc_score(y_valid,gb_model.predict_proba(X_valid)[:, 1])
print("AUC for Gradient Boost: {:.6f}".format(gb_auc))

preds_gb = gb_model.predict(X_valid)
confusion = confusion_matrix(y_valid, preds_gb)
print("Confusion matrix:\n{}".format(confusion))
```

    AUC for Gradient Boost: 0.862094
    Confusion matrix:
    [[24870   269]
     [ 1496   365]]
    


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
                                       id_col='Id',
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
rf_model = RandomForestClassifier(n_estimators=300,random_state=10, n_jobs=-1).fit(X_train,y_train)
rf_auc = roc_auc_score(y_valid,rf_model.predict_proba(X_valid)[:, 1])
print("AUC for Random Forest: {:.6f}".format(rf_auc))
preds_rf = rf_model.predict(X_valid)
confusion = confusion_matrix(y_valid, preds_rf)
print("Confusion matrix:\n{}".format(confusion))
```

    AUC for Random Forest: 0.837342
    Confusion matrix:
    [[24837   302]
     [ 1521   340]]
    


```python
gb_model = GradientBoostingClassifier(n_estimators=300,random_state=10).fit(X_train,y_train)
gb_auc = roc_auc_score(y_valid,gb_model.predict_proba(X_valid)[:, 1])
print("AUC for Gradient Boost: {:.6f}".format(gb_auc))

preds_gb = gb_model.predict(X_valid)
confusion = confusion_matrix(y_valid, preds_gb)
print("Confusion matrix:\n{}".format(confusion))
```

    AUC for Gradient Boost: 0.861899
    Confusion matrix:
    [[24876   263]
     [ 1490   371]]
    


```python

```
