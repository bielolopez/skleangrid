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
                                       id_col='Id',
                                       log_y=True,
                                       convert_to_cat_cols=['CentralAir','MoSold','YrSold','YearBuilt','GarageYrBlt','YearRemodAdd'],
                                       remove_skewness=True,
                                       scale_mapper=RobustScaler(),
                                       bin_columns_dict={'OverallQual':(1,11,10),'OverallCond':(1,11,10)}
                                       
                                       )

print(combined.shape,len(cat_cols),len(num_cols),n_train,n_test)
```

    0.0
    (2919, 659) 50 29 1460 1459
    


```python
df = combined[:n_train]
df_test = combined[n_train:]
X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=stratify_col,shuffle = True,random_state=20)

rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model,X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.003020649665225961, MSE Validation set = 0.021615481656724885, score Training Set = 0.981065019479805, score on Validation Set = 0.8640540653629617
    OOB Score = 0.8592600698956999
    


```python
print_feature_importances(rf_model,X_train)
```

    GrLivArea                     0.313838
    ExterQual_TA                  0.221572
    TotalBsmtSF                   0.069519
    GarageCars                    0.051918
    GarageArea                    0.043766
    1stFlrSF                      0.028304
    BsmtFinSF1                    0.017738
    LotArea                       0.016404
    CentralAir_Y                  0.014887
    CentralAir_N                  0.013762
    2ndFlrSF                      0.010852
    FullBath                      0.008507
    ExterQual_Fa                  0.008233
    BsmtQual_Ex                   0.007386
    MSZoning_C (all)              0.006708
    BsmtUnfSF                     0.005532
    KitchenQual_TA                0.005258
    LotFrontage                   0.005194
    OpenPorchSF                   0.004597
    OverallQual_(3.222, 4.333]    0.004113
    TotRmsAbvGrd                  0.003767
    GarageType_Attchd             0.003763
    KitchenQual_Gd                0.003603
    HalfBath                      0.003550
    BsmtQual_Gd                   0.003460
    ExterQual_Gd                  0.003429
    MSSubClass                    0.003281
    MasVnrArea                    0.003145
    KitchenQual_Ex                0.003131
    WoodDeckSF                    0.002808
    BedroomAbvGr                  0.002660
    KitchenAbvGr                  0.002573
    Fireplaces                    0.002384
    GarageQual_TA                 0.002274
    MSZoning_RM                   0.002263
    EnclosedPorch                 0.002193
    YearRemodAdd_1950             0.002151
    OverallQual_(6.556, 7.667]    0.002147
    OverallCond_(2.111, 3.222]    0.002144
    ExterCond_Fa                  0.002032
    OverallQual_(4.333, 5.444]    0.002028
    GarageCond_TA                 0.001942
    GarageFinish_Unf              0.001923
    KitchenQual_Fa                0.001911
    OverallQual_(7.667, 8.778]    0.001906
    PavedDrive_N                  0.001823
    FireplaceQu_nan               0.001781
    BsmtQual_TA                   0.001392
    MSZoning_RL                   0.001370
    OverallQual_(8.778, 9.889]    0.001158
    dtype: float64
    


```python
df_raw = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)
df = df_raw.copy()
stratify_col = df['OverallQual'].copy()
combined,df,y,cat_cols,num_cols,test_id,n_train,n_test = preprocess_df(
                                       df_train=df,df_test=df_test,
                                       drop_target=True,
                                       new_features_func=add_new_features1,
                                       id_col='Id',
                                       log_y=True,
                                       convert_to_cat_cols=['CentralAir','MoSold','YrSold','YearBuilt','GarageYrBlt','YearRemodAdd'],
                                       remove_skewness=True,
                                       scale_mapper=RobustScaler(),
                                       bin_columns_dict={'OverallQual':(1,11,10),'OverallCond':(1,11,10)}
                                       )

print(combined.shape,len(cat_cols),len(num_cols),n_train,n_test)
```

    0.0
    (2919, 661) 50 30 1460 1459
    


```python
df = combined[:n_train]
df_test = combined[n_train:]
X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=stratify_col,shuffle = True,random_state=20)
rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model, X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.002954822171035181, MSE Validation set = 0.021136202091521436, score Training Set = 0.9814776599572973, score on Validation Set = 0.8670683913668304
    OOB Score = 0.8633395137532879
    


```python
df_raw = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)
df = df_raw.copy()
stratify_col = df['OverallQual'].copy()
combined,df,y,cat_cols,num_cols,test_id,n_train,n_test = preprocess_df(
                                       df_train=df,df_test=df_test,
                                       drop_target=True,
                                       new_features_func=add_new_features2,
                                       id_col='Id',
                                       log_y=True,
                                       convert_to_cat_cols=['CentralAir','MoSold','YrSold','YearBuilt','GarageYrBlt','YearRemodAdd'],
                                       remove_skewness=True,
                                       scale_mapper=RobustScaler(),
                                       bin_columns_dict={'OverallQual':(1,11,10),'OverallCond':(1,11,10)}
                                       )

print(combined.shape,len(cat_cols),len(num_cols),n_train,n_test)
```

    0.0
    (2919, 662) 50 31 1460 1459
    


```python
df = combined[:n_train]
df_test = combined[n_train:]
X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=stratify_col,shuffle = True,random_state=20)
rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model, X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.0026960610919114407, MSE Validation set = 0.018722746228316493, score Training Set = 0.9830997070450471, score on Validation Set = 0.882247304251548
    OOB Score = 0.8754172131364143
    


```python
print_feature_importances(rf_model,X_train)
```

    TotalSF                       0.614958
    OverallGrade                  0.068104
    ExterQual_TA                  0.054294
    GarageCars                    0.032548
    GarageArea                    0.016599
    KitchenQual_TA                0.012358
    LotArea                       0.011558
    GrLivArea                     0.010779
    2ndFlrSF                      0.008854
    BsmtFinSF1                    0.008641
    CentralAir_Y                  0.007003
    BsmtUnfSF                     0.006975
    CentralAir_N                  0.006550
    1stFlrSF                      0.006498
    FullBath                      0.005710
    TotalBsmtSF                   0.004523
    MSZoning_C (all)              0.004001
    LotFrontage                   0.003843
    BsmtQual_Gd                   0.003521
    KitchenQual_Gd                0.003519
    BsmtQual_Ex                   0.003299
    OpenPorchSF                   0.003198
    MSSubClass                    0.002688
    GarageFinish_Unf              0.002620
    Foundation_PConc              0.002576
    Neighborhood_OldTown          0.002565
    BsmtQual_TA                   0.002425
    WoodDeckSF                    0.002397
    MasVnrArea                    0.002388
    TotRmsAbvGrd                  0.002294
    ExterQual_Gd                  0.002282
    OverallQual_(6.556, 7.667]    0.002138
    KitchenQual_Ex                0.001729
    MSZoning_RM                   0.001516
    EnclosedPorch                 0.001471
    KitchenAbvGr                  0.001370
    GarageType_Detchd             0.001321
    BedroomAbvGr                  0.001287
    SaleCondition_Abnorml         0.001283
    GarageType_Attchd             0.001231
    KitchenQual_Fa                0.001230
    LandContour_Bnk               0.001217
    YearRemodAdd_1950             0.001180
    MSZoning_RL                   0.001072
    ExterCond_Fa                  0.001061
    OverallQual_(4.333, 5.444]    0.001051
    PavedDrive_N                  0.000978
    OverallQual_(3.222, 4.333]    0.000972
    Neighborhood_Crawfor          0.000962
    Fireplaces                    0.000942
    dtype: float64
    


```python
df_raw = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)
df = df_raw.copy()
stratify_col = df['OverallQual'].copy()
combined,df,y,cat_cols,num_cols,test_id,n_train,n_test = preprocess_df(
                                       df_train=df,df_test=df_test,
                                       drop_target=True,
                                       new_features_func=add_new_features2,
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
X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=stratify_col,shuffle = True,random_state=20)

rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model, X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.002710722156369716, MSE Validation set = 0.01900308764016412, score Training Set = 0.9830078039775982, score on Validation Set = 0.8804841570843308
    OOB Score = 0.873938823728384
    


```python
print_feature_importances(rf_model,X_train)
```

    TotalSF                       0.612884
    OverallGrade                  0.068920
    ExterQual_TA                  0.053414
    GarageCars_2.0                0.022094
    GarageCars_3.0                0.016273
    GarageArea                    0.015899
    KitchenQual_TA                0.011959
    LotArea                       0.011512
    GrLivArea                     0.010784
    2ndFlrSF                      0.008842
    BsmtFinSF1                    0.008568
    BsmtUnfSF                     0.007074
    1stFlrSF                      0.006563
    CentralAir_Y                  0.005918
    CentralAir_N                  0.005778
    FullBath                      0.005030
    TotalBsmtSF                   0.004797
    MSZoning_C (all)              0.003967
    LotFrontage                   0.003696
    KitchenQual_Gd                0.003301
    OpenPorchSF                   0.003251
    BsmtQual_Gd                   0.003132
    BsmtQual_TA                   0.002953
    BsmtQual_Ex                   0.002856
    GarageFinish_Unf              0.002763
    Foundation_PConc              0.002550
    MSSubClass                    0.002493
    Neighborhood_OldTown          0.002458
    ExterQual_Gd                  0.002454
    MasVnrArea                    0.002436
    WoodDeckSF                    0.002377
    TotRmsAbvGrd                  0.002206
    OverallQual_(6.556, 7.667]    0.002029
    KitchenAbvGr                  0.001575
    EnclosedPorch                 0.001446
    MSZoning_RM                   0.001441
    KitchenQual_Ex                0.001381
    BedroomAbvGr                  0.001316
    GarageType_Detchd             0.001314
    KitchenQual_Fa                0.001308
    GarageType_Attchd             0.001273
    OverallQual_(4.333, 5.444]    0.001225
    MSZoning_RL                   0.001210
    YearRemodAdd_1950             0.001171
    OverallQual_(3.222, 4.333]    0.001163
    SaleCondition_Abnorml         0.001150
    PavedDrive_N                  0.001136
    ExterCond_Fa                  0.001049
    LandContour_Bnk               0.000999
    Fireplaces                    0.000969
    dtype: float64
    


```python
df_raw = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)
df = df_raw.copy()
stratify_col = df['OverallQual'].copy()
combined,df,y,cat_cols,num_cols,test_id,n_train,n_test = preprocess_df(
                                       df_train=df,df_test=df_test,
                                       drop_target=True,
                                       new_features_func=add_new_features3,
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
X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=stratify_col,shuffle = True,random_state=20)

rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model, X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.0026983463732659875, MSE Validation set = 0.019063615989086742, score Training Set = 0.9830853817300563, score on Validation Set = 0.8801034770191348
    OOB Score = 0.8756880077529818
    


```python
print_feature_importances(rf_model,X_train)
```

    TotalSF                       0.611868
    OverallGrade                  0.066087
    ExterQual_TA                  0.052197
    GarageCars_2.0                0.022281
    GarageCars_3.0                0.016748
    GarageArea                    0.016249
    TotalLivArea                  0.013068
    KitchenQual_TA                0.012786
    GrLivArea                     0.010250
    2ndFlrSF                      0.009045
    BsmtFinSF1                    0.008224
    BsmtUnfSF                     0.006654
    CentralAir_Y                  0.006585
    CentralAir_N                  0.006364
    1stFlrSF                      0.006052
    FullBath                      0.004926
    TotalBsmtSF                   0.004649
    LotArea                       0.004370
    MSZoning_C (all)              0.004038
    LotFrontage                   0.003485
    BsmtQual_Gd                   0.003349
    OpenPorchSF                   0.003213
    BsmtQual_TA                   0.003057
    GarageFinish_Unf              0.002931
    BsmtQual_Ex                   0.002789
    KitchenQual_Gd                0.002756
    MSSubClass                    0.002652
    Neighborhood_OldTown          0.002398
    Foundation_PConc              0.002394
    WoodDeckSF                    0.002362
    MasVnrArea                    0.002344
    OverallQual_(6.556, 7.667]    0.002322
    TotRmsAbvGrd                  0.002286
    ExterQual_Gd                  0.002274
    MSZoning_RM                   0.001542
    GarageType_Attchd             0.001537
    KitchenQual_Ex                0.001411
    EnclosedPorch                 0.001390
    KitchenAbvGr                  0.001369
    KitchenQual_Fa                0.001362
    BedroomAbvGr                  0.001253
    GarageType_Detchd             0.001223
    SaleCondition_Abnorml         0.001215
    PavedDrive_N                  0.001108
    OverallQual_(4.333, 5.444]    0.001103
    OverallQual_(3.222, 4.333]    0.001102
    LandContour_Bnk               0.001081
    YearRemodAdd_1950             0.001039
    ExterCond_Fa                  0.001036
    GarageQual_TA                 0.000982
    dtype: float64
    


```python
df_raw = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)
df = df_raw.copy()
stratify_col = df['OverallQual'].copy()
combined,df,y,cat_cols,num_cols,_,n_train,n_test = preprocess_df(
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
X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=stratify_col,shuffle = True,random_state=20)

rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model, X_train,X_valid,y_train,y_valid)
```

    MSE Training set = 0.0026919425996773097, MSE Validation set = 0.019061215547191975, score Training Set = 0.9831255238655555, score on Validation Set = 0.880118574083457
    OOB Score = 0.8745363168334529
    


```python
print_feature_importances(rf_model,X_train)
```

    TotalSF                       0.611979
    OverallGrade                  0.066045
    ExterQual_TA                  0.052285
    GarageCars_2.0                0.022202
    GarageCars_3.0                0.015046
    KitchenQual_TA                0.013428
    TotalLivArea                  0.012791
    2ndFlrSF                      0.008789
    BsmtFinSF1                    0.008269
    BsmtUnfSF                     0.006745
    CentralAir_Y                  0.006394
    CentralAir_N                  0.006281
    1stFlrSF                      0.006073
    FullBath                      0.004795
    TotalBsmtSF                   0.004616
    GarageArea-Sq                 0.004507
    GarageArea-2                  0.004449
    LotArea                       0.004193
    GarageArea                    0.004168
    MSZoning_C (all)              0.004056
    GarageArea-3                  0.004038
    BsmtQual_Ex                   0.003580
    LotFrontage                   0.003350
    GrLivArea-2                   0.003171
    OpenPorchSF                   0.003137
    GrLivArea-3                   0.003083
    GrLivArea                     0.003062
    BsmtQual_TA                   0.003033
    BsmtQual_Gd                   0.003012
    GrLivArea-Sq                  0.002961
    GarageFinish_Unf              0.002770
    ExterQual_Gd                  0.002732
    KitchenQual_Gd                0.002675
    Foundation_PConc              0.002581
    MSSubClass                    0.002529
    Neighborhood_OldTown          0.002387
    MasVnrArea                    0.002346
    WoodDeckSF                    0.002279
    OverallQual_(6.556, 7.667]    0.002268
    TotRmsAbvGrd                  0.002194
    KitchenQual_Ex                0.001468
    GarageType_Attchd             0.001457
    KitchenAbvGr                  0.001398
    MSZoning_RM                   0.001369
    EnclosedPorch                 0.001365
    BedroomAbvGr                  0.001276
    KitchenQual_Fa                0.001274
    SaleCondition_Abnorml         0.001226
    GarageType_Detchd             0.001207
    YearRemodAdd_1950             0.001157
    dtype: float64
    


```python

```


```python

```
