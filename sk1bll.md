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

```



```python
df_raw = pd.read_csv('train.csv', low_memory=False)
df_test = pd.read_csv('test.csv', low_memory=False)

```


```python
df_test.shape,df_raw.shape
```




    ((1459, 80), (1460, 81))




```python
df_raw.info()
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>...</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>730.500000</td>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>...</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>421.610009</td>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>...</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>365.750000</td>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>730.500000</td>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1095.250000</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1460.000000</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>...</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 38 columns</p>
</div>




```python
df = df_raw.copy()
```


```python
df,y = extract_and_drop_target_column(df,'SalePrice',inplace=True)
```


```python
ntrain = df.shape[0]
ntest = df_test.shape[0]
test_id = df_test['Id'].copy()
```


```python
combined = pd.concat((df, df_test)).reset_index(drop=True)
```


```python
combined.drop('Id', axis=1,inplace=True)
```


```python
cat_cols = get_cat_columns_by_type(combined)
```


```python
print(cat_cols)
```

    [('MSZoning', 'string'), ('Street', 'string'), ('Alley', 'string'), ('LotShape', 'string'), ('LandContour', 'string'), ('Utilities', 'string'), ('LotConfig', 'string'), ('LandSlope', 'string'), ('Neighborhood', 'string'), ('Condition1', 'string'), ('Condition2', 'string'), ('BldgType', 'string'), ('HouseStyle', 'string'), ('RoofStyle', 'string'), ('RoofMatl', 'string'), ('Exterior1st', 'string'), ('Exterior2nd', 'string'), ('MasVnrType', 'string'), ('ExterQual', 'string'), ('ExterCond', 'string'), ('Foundation', 'string'), ('BsmtQual', 'string'), ('BsmtCond', 'string'), ('BsmtExposure', 'string'), ('BsmtFinType1', 'string'), ('BsmtFinType2', 'string'), ('Heating', 'string'), ('HeatingQC', 'string'), ('CentralAir', 'string'), ('Electrical', 'string'), ('KitchenQual', 'string'), ('Functional', 'string'), ('FireplaceQu', 'string'), ('GarageType', 'string'), ('GarageFinish', 'string'), ('GarageQual', 'string'), ('GarageCond', 'string'), ('PavedDrive', 'string'), ('PoolQC', 'string'), ('Fence', 'string'), ('MiscFeature', 'string'), ('SaleType', 'string'), ('SaleCondition', 'string')]
    


```python
len(cat_cols)
```




    43




```python
cat_cols = [cat_cols[i][0] for i in range(len(cat_cols))]
cat_cols
```




    ['MSZoning',
     'Street',
     'Alley',
     'LotShape',
     'LandContour',
     'Utilities',
     'LotConfig',
     'LandSlope',
     'Neighborhood',
     'Condition1',
     'Condition2',
     'BldgType',
     'HouseStyle',
     'RoofStyle',
     'RoofMatl',
     'Exterior1st',
     'Exterior2nd',
     'MasVnrType',
     'ExterQual',
     'ExterCond',
     'Foundation',
     'BsmtQual',
     'BsmtCond',
     'BsmtExposure',
     'BsmtFinType1',
     'BsmtFinType2',
     'Heating',
     'HeatingQC',
     'CentralAir',
     'Electrical',
     'KitchenQual',
     'Functional',
     'FireplaceQu',
     'GarageType',
     'GarageFinish',
     'GarageQual',
     'GarageCond',
     'PavedDrive',
     'PoolQC',
     'Fence',
     'MiscFeature',
     'SaleType',
     'SaleCondition']




```python
num_cols = [col for col in combined.columns if col not in cat_cols]
len(num_cols)
```




    36




```python
num_cols = get_numeric_columns(combined)
len(num_cols)
```




    36




```python
combined.shape
```




    (2919, 79)




```python
df_raw.shape
```




    (1460, 81)




```python
combined.head()
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
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 79 columns</p>
</div>




```python
combined,d = handle_missing_values(combined,cat_cols=cat_cols,num_cols=num_cols,inplace=True)
d
```




    {'BsmtFinSF1': 368.5,
     'BsmtFinSF2': 0.0,
     'BsmtFullBath': 0.0,
     'BsmtHalfBath': 0.0,
     'BsmtUnfSF': 467.0,
     'GarageArea': 480.0,
     'GarageCars': 2.0,
     'GarageYrBlt': 1979.0,
     'LotFrontage': 68.0,
     'MasVnrArea': 0.0,
     'TotalBsmtSF': 989.5}




```python
get_missing_values_percentage(combined)
```




    0.0




```python
df = combined[:ntrain]
df_test = combined[ntrain:]
```


```python
df.shape,df_test.shape
```




    ((1460, 90), (1459, 90))




```python
get_missing_values_percentage(df)
```




    0.0




```python
os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/raw')
```


```python
df_raw = pd.read_feather('tmp/raw')
```


```python
X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=df['OverallQual'],shuffle = True,random_state=20)

```


```python
y.shape,df.shape,df_test.shape
```




    ((1460,), (1460, 90), (1459, 90))




```python
def print_mse(m):
    res = [mean_squared_error(y_train,m.predict(X_train)),
                mean_squared_error(y_valid,m.predict(X_valid)),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    print('MSE Training set = {}, MSE Validation set = {}, score Training Set = {}, score on Validation Set = {}'.format(res[0],res[1],res[2], res[3]))
    if hasattr(m, 'oob_score_'):
          print('OOB Score = {}'.format(m.oob_score_))      
```


```python
rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
```


```python
print_mse(rf_model)
```

    MSE Training set = 117818388.99578248, MSE Validation set = 675499663.2858182, score Training Set = 0.981340154750642, score on Validation Set = 0.8921874142070969
    OOB Score = 0.8609853153379634
    


```python
combined,_ = scale_num_cols(combined,None)
X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.15,
                                  stratify=df['OverallQual'],shuffle = True,random_state=20)
```


```python
rf_model = RandomForestRegressor(n_estimators=1500,n_jobs=-1,oob_score=True).fit(X_train.values,
                                                                                y_train)
print_mse(rf_model)
```

    MSE Training set = 118976934.93358955, MSE Validation set = 671471458.6815141, score Training Set = 0.9811566665184737, score on Validation Set = 0.8928303325949173
    OOB Score = 0.8598427216112249
    


```python

```
