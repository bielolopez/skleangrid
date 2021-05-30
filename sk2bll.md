```python
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error,mean_squared_log_error,r2_score
from sklearn_pandas import DataFrameMapper
from pandas.api.types import is_string_dtype, is_numeric_dtype

from scipy.stats import skew
from scipy.special import boxcox1p

%matplotlib inline
```


```python
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
                 bin_columns_dict=None):
    
    if drop_target:
        df,y = extract_and_drop_target_column(df_train,'SalePrice',inplace=True)
    if log_y:
        y = np.log1p(y)
    else:
        y = None
        
    combined = pd.concat((df, df_test)).reset_index(drop=True)
    
    if df_test is not None and id_col is not None:
        test_id = df_test['Id'].copy()
        combined.drop('Id', axis=1,inplace=True)
    else: test_id = None
    
    if convert_to_cat_cols is not None:
        combined = convert_to_str_type(combined,convert_to_cat_cols)
    if bin_columns_dict is not None:
        combined = bin_numerical_columns(combined,bin_columns_dict,inplace=True)

    cat_cols = get_cat_columns_by_type(combined)
    cat_cols = [cat_cols[i][0] for i in range(len(cat_cols))]
    num_cols = [col for col in combined.columns if col not in cat_cols]
    n_train = df.shape[0]
    n_test = df_test.shape[0]
   
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
      
    combined,d = handle_missing_values(combined,cat_cols=cat_cols,num_cols=num_cols,inplace=True)
    
    combined = pd.get_dummies(combined, dummy_na=True)
    
    if scale_mapper is not None:
        map_f = [([c],scale_mapper) for c in num_cols]
        mapper = DataFrameMapper(map_f).fit(combined)
    else:
        mapper = None
        
    combined,_ = scale_num_cols(combined,mapper) 
    
    return combined,df,y,cat_cols,num_cols,test_id,n_train,n_test

```


```python
df_raw = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)
df = df_raw.copy()
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object 
     6   Alley          91 non-null     object 
     7   LotShape       1460 non-null   object 
     8   LandContour    1460 non-null   object 
     9   Utilities      1460 non-null   object 
     10  LotConfig      1460 non-null   object 
     11  LandSlope      1460 non-null   object 
     12  Neighborhood   1460 non-null   object 
     13  Condition1     1460 non-null   object 
     14  Condition2     1460 non-null   object 
     15  BldgType       1460 non-null   object 
     16  HouseStyle     1460 non-null   object 
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object 
     22  RoofMatl       1460 non-null   object 
     23  Exterior1st    1460 non-null   object 
     24  Exterior2nd    1460 non-null   object 
     25  MasVnrType     1452 non-null   object 
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object 
     28  ExterCond      1460 non-null   object 
     29  Foundation     1460 non-null   object 
     30  BsmtQual       1423 non-null   object 
     31  BsmtCond       1423 non-null   object 
     32  BsmtExposure   1422 non-null   object 
     33  BsmtFinType1   1423 non-null   object 
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object 
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object 
     40  HeatingQC      1460 non-null   object 
     41  CentralAir     1460 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object 
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object 
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object 
     58  GarageType     1379 non-null   object 
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object 
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object 
     64  GarageCond     1379 non-null   object 
     65  PavedDrive     1460 non-null   object 
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object 
     73  Fence          281 non-null    object 
     74  MiscFeature    54 non-null     object 
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object 
     79  SaleCondition  1460 non-null   object 
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB
    


```python
combined,df,y,cat_cols,num_cols,test_id,n_train,n_test = preprocess_df(df_train=df,df_test=df_test,id_col='Id',
                                                                      convert_to_cat_cols=['MoSold','YrSold','YearBuilt','GarageYrBlt','YearRemodAdd'],
                                                                      remove_skewness=False,
                                                                      )
print(combined.shape,len(cat_cols),len(num_cols),n_train,n_test)
```

    (2919, 89) 48 31 1460 1459
    


```python
cat_cols = get_cat_columns_by_type(df_raw)
cat_cols = [cat_cols[i][0] for i in range(len(cat_cols))]
num_cols = [col for col in df_raw.columns if col not in cat_cols]
skewed_cols = df_raw[num_cols].apply(lambda x: skew(x)).sort_values(ascending=False)
skewed_cols
```




    MiscVal          24.451640
    PoolArea         14.813135
    LotArea          12.195142
    3SsnPorch        10.293752
    LowQualFinSF      9.002080
    KitchenAbvGr      4.483784
    BsmtFinSF2        4.250888
    ScreenPorch       4.117977
    BsmtHalfBath      4.099186
    EnclosedPorch     3.086696
    OpenPorchSF       2.361912
    SalePrice         1.880941
    BsmtFinSF1        1.683771
    WoodDeckSF        1.539792
    TotalBsmtSF       1.522688
    MSSubClass        1.406210
    1stFlrSF          1.375342
    GrLivArea         1.365156
    BsmtUnfSF         0.919323
    2ndFlrSF          0.812194
    OverallCond       0.692355
    TotRmsAbvGrd      0.675646
    HalfBath          0.675203
    Fireplaces        0.648898
    BsmtFullBath      0.595454
    OverallQual       0.216721
    MoSold            0.211835
    BedroomAbvGr      0.211572
    GarageArea        0.179796
    YrSold            0.096170
    FullBath          0.036524
    Id                0.000000
    GarageCars       -0.342197
    YearRemodAdd     -0.503044
    YearBuilt        -0.612831
    LotFrontage            NaN
    MasVnrArea             NaN
    GarageYrBlt            NaN
    dtype: float64




```python
iqr_ranges = get_iqr_min_max(df,num_cols)
iqr_ranges
```




    {'1stFlrSF': (118.125, 2155.125),
     '2ndFlrSF': (-1092.0, 1820.0),
     '3SsnPorch': (0.0, 0.0),
     'BedroomAbvGr': (0.5, 4.5),
     'BsmtFinSF1': (-1068.375, 1780.625),
     'BsmtFinSF2': (0.0, 0.0),
     'BsmtFullBath': (-1.5, 2.5),
     'BsmtHalfBath': (0.0, 0.0),
     'BsmtUnfSF': (-654.5, 1685.5),
     'EnclosedPorch': (0.0, 0.0),
     'Fireplaces': (-1.5, 2.5),
     'FullBath': (-0.5, 3.5),
     'GarageArea': (-27.75, 938.25),
     'GarageCars': (-0.5, 3.5),
     'GarageYrBlt': (nan, nan),
     'GrLivArea': (158.625, 2747.625),
     'HalfBath': (-1.5, 2.5),
     'Id': (-728.5, 2189.5),
     'KitchenAbvGr': (1.0, 1.0),
     'LotArea': (1481.5, 17673.5),
     'LotFrontage': (nan, nan),
     'LowQualFinSF': (0.0, 0.0),
     'MSSubClass': (-55.0, 145.0),
     'MasVnrArea': (nan, nan),
     'MiscVal': (0.0, 0.0),
     'MoSold': (0.5, 12.5),
     'OpenPorchSF': (-102.0, 170.0),
     'OverallCond': (3.5, 7.5),
     'OverallQual': (2.0, 10.0),
     'PoolArea': (0.0, 0.0),
     'ScreenPorch': (0.0, 0.0),
     'TotRmsAbvGrd': (2.0, 10.0),
     'TotalBsmtSF': (42.0, 2052.0),
     'WoodDeckSF': (-252.0, 420.0),
     'YearBuilt': (1885.0, 2069.0),
     'YearRemodAdd': (1911.5, 2059.5),
     'YrSold': (2004.0, 2012.0)}




```python
df['GrLivArea'].describe()
```




    count    1460.000000
    mean     1515.463699
    std       525.480383
    min       334.000000
    25%      1129.500000
    50%      1464.000000
    75%      1776.750000
    max      5642.000000
    Name: GrLivArea, dtype: float64




```python
df['GrLivArea'].median()
```




    1464.0




```python
df = combined[:n_train]
df_test = combined[n_train:]

X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=df['OverallQual'],shuffle = True,random_state=20)
print('Data Ready')
```

    Data Ready
    


```python
rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model,X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.0027803797301694903, MSE Validation set = 0.01881236964293611, score Training Set = 0.9825711545977756, score on Validation Set = 0.8816836370125142
    OOB Score = 0.8726454952639682
    


```python
df_raw = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)
df = df_raw.copy()
combined,df,y,cat_cols,num_cols,_,n_train,n_test = preprocess_df(df_train=df,df_test=df_test,
                                                                       drop_target=True,
                                                                       id_col='Id',
                                                                       log_y=True,
                                                                       convert_to_cat_cols=['MoSold','YrSold','YearBuilt','GarageYrBlt','YearRemodAdd'],
                                                                       remove_skewness=True
                                                                       )
print(combined.shape,len(cat_cols),len(num_cols),n_train,n_test)
```

    (2919, 89) 48 31 1460 1459
    


```python
df = combined[:n_train]
df_test = combined[n_train:]

X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=df['OverallQual'],shuffle = True,random_state=20)
print('Data Ready')
```

    Data Ready
    


```python
rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model,X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.0027415963324854092, MSE Validation set = 0.018674822941861402, score Training Set = 0.9828142688152597, score on Validation Set = 0.8825487074805614
    OOB Score = 0.8724261781463059
    


```python
df_raw = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)
df = df_raw.copy()
combined,df,y,cat_cols,num_cols,_,n_train,n_test = preprocess_df(df_train=df,df_test=df_test,
                                                   drop_target=True,
                                                   id_col='Id',
                                                   log_y=True,
                                                   convert_to_cat_cols=['MoSold','YrSold','YearBuilt','GarageYrBlt','YearRemodAdd'],
                                                   remove_skewness=True,
                                                   scale_mapper=RobustScaler()
                                                  )
```


```python
df = combined[:n_train]
df_test = combined[n_train:]

X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=df['OverallQual'],shuffle = True,random_state=20)
print('Data Ready')
```

    Data Ready
    


```python
rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model,X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.0027514247108097285, MSE Validation set = 0.018630987110396614, score Training Set = 0.982752659501787, score on Validation Set = 0.8828244035383088
    OOB Score = 0.872493304068987
    


```python
important_features = pd.Series(data=rf_model.feature_importances_,index=X_train.columns)
important_features.sort_values(ascending=False,inplace=True)
important_features.head(50)
```




    OverallQual      0.545199
    GrLivArea        0.115615
    TotalBsmtSF      0.049093
    GarageArea       0.035058
    GarageCars       0.034601
    1stFlrSF         0.020321
    BsmtFinSF1       0.019873
    YearBuilt        0.014800
    LotArea          0.012748
    OverallCond      0.012344
    CentralAir       0.011602
    MSZoning         0.009112
    YearRemodAdd     0.007840
    2ndFlrSF         0.007360
    GarageFinish     0.007330
    BsmtUnfSF        0.005538
    Neighborhood     0.004980
    LotFrontage      0.004625
    Fireplaces       0.004341
    GarageYrBlt      0.004283
    OpenPorchSF      0.003837
    MoSold           0.003493
    GarageType       0.003346
    FullBath         0.003236
    WoodDeckSF       0.003220
    MasVnrArea       0.003152
    TotRmsAbvGrd     0.002933
    EnclosedPorch    0.002733
    BsmtQual         0.002723
    ExterCond        0.002481
    KitchenQual      0.002454
    BsmtFinType1     0.002304
    BedroomAbvGr     0.002173
    SaleCondition    0.002133
    MSSubClass       0.002096
    LandContour      0.001903
    LotShape         0.001861
    Exterior1st      0.001810
    ExterQual        0.001745
    BsmtExposure     0.001714
    YrSold           0.001665
    Exterior2nd      0.001642
    PavedDrive       0.001601
    HeatingQC        0.001113
    HouseStyle       0.000916
    BsmtFullBath     0.000890
    LandSlope        0.000885
    HalfBath         0.000817
    Foundation       0.000809
    RoofStyle        0.000796
    dtype: float64




```python
combined['CentralAir'].value_counts()
```




    1    2723
    0     196
    Name: CentralAir, dtype: int64




```python
combined['OverallQual'].value_counts()
```




    1.109946    825
    1.172893    731
    1.225196    600
    1.269755    342
    1.031562    226
    1.308446    107
    0.929025     40
    1.342554     31
    0.784059     13
    0.547945      4
    Name: OverallQual, dtype: int64




```python
combined['OverallCond'].value_counts()
```




    1.109946    1645
    1.172893     531
    1.225196     390
    1.269755     144
    1.031562     101
    0.929025      50
    1.308446      41
    0.784059      10
    0.547945       7
    Name: OverallCond, dtype: int64




```python
df_raw = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)
df = df_raw.copy()
combined,df,y,cat_cols,num_cols,_,n_train,n_test = preprocess_df(df_train=df,df_test=df_test,
                                                                       drop_target=True,
                                                                       id_col='Id',
                                                                       log_y=True,
                                                                       convert_to_cat_cols=['CentralAir','MoSold','YrSold','YearBuilt','GarageYrBlt','YearRemodAdd'],
                                                                       remove_skewness=True,
                                                                       scale_mapper=RobustScaler(),
                                                                       bin_columns_dict={'OverallQual':(1,11,10),'OverallCond':(1,11,10)}
                                                                   )


```


```python
df = combined[:n_train]
df_test = combined[n_train:]

X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=df['OverallQual'],shuffle = True,random_state=20)
rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model,X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.0027054575448710923, MSE Validation set = 0.02035193610834659, score Training Set = 0.9828503943543676, score on Validation Set = 0.8795820639749579
    OOB Score = 0.8729929249489109
    


```python
df_raw = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)
df = df_raw.copy()
combined,df,y,cat_cols,num_cols,test_id,n_train,n_test = preprocess_df(df_train=df,df_test=df_test,
                                                                       drop_target=True,
                                                                       id_col='Id',
                                                                       log_y=True,
                                                                       convert_to_cat_cols=['MoSold','YrSold','YearBuilt','GarageYrBlt','YearRemodAdd'],
                                                                       remove_skewness=True,
                                                                       scale_mapper=RobustScaler(),
                                                                       bin_columns_dict={'OverallQual':(1,11,10),'OverallCond':(1,11,10)}
                                                                   )


map_f = [([c],MinMaxScaler()) for c in combined.columns if is_numeric_dtype(combined[c])]
mapper = DataFrameMapper(map_f).fit(combined)
combined,_ = scale_num_cols(combined,mapper)
```


```python
df = combined[:n_train]
df_test = combined[n_train:]

X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=df['OverallQual'],shuffle = True,random_state=20)
rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model,X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.002759138692459974, MSE Validation set = 0.020602868725358676, score Training Set = 0.9825101153085923, score on Validation Set = 0.8780973507928262
    OOB Score = 0.8718095316751527
    


```python
df_raw = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)
df = df_raw.copy()
combined,df,y,cat_cols,num_cols,test_id,n_train,n_test = preprocess_df(df_train=df,df_test=df_test,
                                                                       drop_target=True,
                                                                       id_col='Id',
                                                                       log_y=True,
                                                                       convert_to_cat_cols=['MoSold','YrSold','YearBuilt','GarageYrBlt','YearRemodAdd'],
                                                                       remove_skewness=True,
                                                                       scale_mapper=RobustScaler(),
                                                                       bin_columns_dict={'OverallQual':(1,11,10),'OverallCond':(1,11,10)}
                                                                   )


combined,_ = scale_num_cols(combined,None)
```


```python
df = combined[:n_train]
df_test = combined[n_train:]

X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=df['OverallQual'],shuffle = True,random_state=20)
rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model,X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.0027427743404250743, MSE Validation set = 0.020393056396343853, score Training Set = 0.9826138471836597, score on Validation Set = 0.8793387642621922
    OOB Score = 0.8723048043910675
    


```python

```
