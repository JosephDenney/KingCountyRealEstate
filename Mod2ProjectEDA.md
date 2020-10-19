# Housing Analysis in King County, Washington
### EDA, data cleaning, feature engineering notebook


```python
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np 
import csv

import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.formula.api as sms
import scipy.stats as stats
from statsFunctions import check_model as sf
from pltfunctions import hist_kde_plots
from haversine import haversine
from math import sqrt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'data\kc_house_data.csv')
```

### Data description as follows
![title](img/headers.png)


```python
df.info() # a good initial picture of the data
# initial thoughts for necessary data and a regression upon seeing the data -
# price is y target
# 1) Unique identifiers (column= id) are unnecessary for a regression
# 2) lat and long wont be needed if include zipcode and will just be noise (decided to keep lat/long over zip)
# 3) square footage and home quality is likely to change based on location
# 4) anticipate autocorrelation between location and home features - dropping zipcode and keeping lat long
# 5) date - as time goes on, home prices are likely to go up - if the sales are all within a close time frame then we can ignore date
# 6) nearest 15 neighbors data - will autocorrelate with homes nearby, remove data
# 7) already have square footage of home, remove above and below ground sqft
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             21597 non-null  int64  
     1   date           21597 non-null  object 
     2   price          21597 non-null  float64
     3   bedrooms       21597 non-null  int64  
     4   bathrooms      21597 non-null  float64
     5   sqft_living    21597 non-null  int64  
     6   sqft_lot       21597 non-null  int64  
     7   floors         21597 non-null  float64
     8   waterfront     19221 non-null  float64
     9   view           21534 non-null  float64
     10  condition      21597 non-null  int64  
     11  grade          21597 non-null  int64  
     12  sqft_above     21597 non-null  int64  
     13  sqft_basement  21597 non-null  object 
     14  yr_built       21597 non-null  int64  
     15  yr_renovated   17755 non-null  float64
     16  zipcode        21597 non-null  int64  
     17  lat            21597 non-null  float64
     18  long           21597 non-null  float64
     19  sqft_living15  21597 non-null  int64  
     20  sqft_lot15     21597 non-null  int64  
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.5+ MB
    


```python
# before we drop any null values (in yr_renovated and waterfront specifically) we should address the issues above.
df_init = df.drop(columns=['id','sqft_living15','sqft_above','sqft_basement']) 
# filter for more recent home sales
df_init['date'].unique() 
# can remove date and null values
print(df['yr_renovated'].unique())
print(df.groupby(df['yr_renovated']).count())
# given that these dates are very spread out and also that 17011 of the homes sold out have no renovation year at all, I will remove the data for the purposes of a linear regression
```

    [   0. 1991.   nan 2002. 2010. 1992. 2013. 1994. 1978. 2005. 2003. 1984.
     1954. 2014. 2011. 1983. 1945. 1990. 1988. 1977. 1981. 1995. 2000. 1999.
     1998. 1970. 1989. 2004. 1986. 2007. 1987. 2006. 1985. 2001. 1980. 1971.
     1979. 1997. 1950. 1969. 1948. 2009. 2015. 1974. 2008. 1968. 2012. 1963.
     1951. 1962. 1953. 1993. 1996. 1955. 1982. 1956. 1940. 1976. 1946. 1975.
     1964. 1973. 1957. 1959. 1960. 1967. 1965. 1934. 1972. 1944. 1958.]
                     id   date  price  bedrooms  bathrooms  sqft_living  sqft_lot  \
    yr_renovated                                                                    
    0.0           17011  17011  17011     17011      17011        17011     17011   
    1934.0            1      1      1         1          1            1         1   
    1940.0            2      2      2         2          2            2         2   
    1944.0            1      1      1         1          1            1         1   
    1945.0            3      3      3         3          3            3         3   
    ...             ...    ...    ...       ...        ...          ...       ...   
    2011.0            9      9      9         9          9            9         9   
    2012.0            8      8      8         8          8            8         8   
    2013.0           31     31     31        31         31           31        31   
    2014.0           73     73     73        73         73           73        73   
    2015.0           14     14     14        14         14           14        14   
    
                  floors  waterfront   view  condition  grade  sqft_above  \
    yr_renovated                                                            
    0.0            17011       15157  16961      17011  17011       17011   
    1934.0             1           1      1          1      1           1   
    1940.0             2           2      2          2      2           2   
    1944.0             1           1      1          1      1           1   
    1945.0             3           2      3          3      3           3   
    ...              ...         ...    ...        ...    ...         ...   
    2011.0             9           7      9          9      9           9   
    2012.0             8           7      8          8      8           8   
    2013.0            31          29     31         31     31          31   
    2014.0            73          64     73         73     73          73   
    2015.0            14          13     14         14     14          14   
    
                  sqft_basement  yr_built  zipcode    lat   long  sqft_living15  \
    yr_renovated                                                                  
    0.0                   17011     17011    17011  17011  17011          17011   
    1934.0                    1         1        1      1      1              1   
    1940.0                    2         2        2      2      2              2   
    1944.0                    1         1        1      1      1              1   
    1945.0                    3         3        3      3      3              3   
    ...                     ...       ...      ...    ...    ...            ...   
    2011.0                    9         9        9      9      9              9   
    2012.0                    8         8        8      8      8              8   
    2013.0                   31        31       31     31     31             31   
    2014.0                   73        73       73     73     73             73   
    2015.0                   14        14       14     14     14             14   
    
                  sqft_lot15  
    yr_renovated              
    0.0                17011  
    1934.0                 1  
    1940.0                 2  
    1944.0                 1  
    1945.0                 3  
    ...                  ...  
    2011.0                 9  
    2012.0                 8  
    2013.0                31  
    2014.0                73  
    2015.0                14  
    
    [70 rows x 20 columns]
    


```python
df_init = df_init.dropna() # drop null values
df_init = df_init.drop(columns=['date','yr_renovated']) # drop date column
```


```python
df_init.head()
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
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1951</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1965</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1987</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>7503</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1230000.0</td>
      <td>4</td>
      <td>4.50</td>
      <td>5420</td>
      <td>101930</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>11</td>
      <td>2001</td>
      <td>98053</td>
      <td>47.6561</td>
      <td>-122.005</td>
      <td>101930</td>
    </tr>
    <tr>
      <th>6</th>
      <td>257500.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>1715</td>
      <td>6819</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1995</td>
      <td>98003</td>
      <td>47.3097</td>
      <td>-122.327</td>
      <td>6819</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_init.info() # data is all in integer or float dtype
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 15762 entries, 1 to 21596
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   price        15762 non-null  float64
     1   bedrooms     15762 non-null  int64  
     2   bathrooms    15762 non-null  float64
     3   sqft_living  15762 non-null  int64  
     4   sqft_lot     15762 non-null  int64  
     5   floors       15762 non-null  float64
     6   waterfront   15762 non-null  float64
     7   view         15762 non-null  float64
     8   condition    15762 non-null  int64  
     9   grade        15762 non-null  int64  
     10  yr_built     15762 non-null  int64  
     11  zipcode      15762 non-null  int64  
     12  lat          15762 non-null  float64
     13  long         15762 non-null  float64
     14  sqft_lot15   15762 non-null  int64  
    dtypes: float64(7), int64(8)
    memory usage: 1.9 MB
    


```python
features = ['bedrooms','bathrooms','sqft_living','sqft_lot','sqft_lot15','floors','waterfront','view','condition','grade','yr_built','yr_renovated','lat','long']
```


```python
for feature in features:
    hist_kde_plots(feature,'price',df_init)
```

    bedrooms
    


    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_1.png)
    



    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_2.png)
    


    bathrooms
    


    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_4.png)
    



    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_5.png)
    


    sqft_living
    


    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_7.png)
    



    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_8.png)
    


    sqft_lot
    


    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_10.png)
    



    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_11.png)
    


    sqft_lot15
    


    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_13.png)
    



    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_14.png)
    


    floors
    


    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_16.png)
    



    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_17.png)
    


    waterfront
    


    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_19.png)
    



    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_20.png)
    


    view
    


    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_22.png)
    



    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_23.png)
    


    condition
    


    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_25.png)
    



    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_26.png)
    


    grade
    


    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_28.png)
    



    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_29.png)
    


    yr_built
    


    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_31.png)
    



    
![png](Mod2ProjectEDA_files/Mod2ProjectEDA_9_32.png)
    


    yr_renovated
    


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    ~\anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       2645             try:
    -> 2646                 return self._engine.get_loc(key)
       2647             except KeyError:
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    KeyError: 'yr_renovated'

    
    During handling of the above exception, another exception occurred:
    

    KeyError                                  Traceback (most recent call last)

    <ipython-input-9-ca5a9d20d9bc> in <module>
          1 for feature in features:
    ----> 2     hist_kde_plots(feature,'price',df_init)
    

    ~\desktop\coursework\phase_1\Phase2\Phase_2_Project\pltfunctions.py in hist_kde_plots(feature, target, df)
         52     print(feature)
         53     #kde plot
    ---> 54     df[feature].plot.kde(label=feature)
         55     plt.title("{} Kde plot".format(feature))
         56     plt.legend()
    

    ~\anaconda3\lib\site-packages\pandas\core\frame.py in __getitem__(self, key)
       2798             if self.columns.nlevels > 1:
       2799                 return self._getitem_multilevel(key)
    -> 2800             indexer = self.columns.get_loc(key)
       2801             if is_integer(indexer):
       2802                 indexer = [indexer]
    

    ~\anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       2646                 return self._engine.get_loc(key)
       2647             except KeyError:
    -> 2648                 return self._engine.get_loc(self._maybe_cast_indexer(key))
       2649         indexer = self.get_indexer([key], method=method, tolerance=tolerance)
       2650         if indexer.ndim > 1 or indexer.size > 1:
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    KeyError: 'yr_renovated'



```python
# funky data in the following columns - 1) yr_renovated - a lot of homes that haven't been
# renovated and they are all listed as 0's. Need to reconcile this data - because of groupby count analysis above decided to just remove this data (<17000 homes were never renovated per original dataset)
# 2) remove any price data that is greater than 3 standard deviations from the mean
df_init.describe()
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
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.576200e+04</td>
      <td>15762.000000</td>
      <td>15762.000000</td>
      <td>15762.000000</td>
      <td>1.576200e+04</td>
      <td>15762.000000</td>
      <td>15762.000000</td>
      <td>15762.000000</td>
      <td>15762.000000</td>
      <td>15762.000000</td>
      <td>15762.000000</td>
      <td>15762.000000</td>
      <td>15762.000000</td>
      <td>15762.000000</td>
      <td>15762.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.413172e+05</td>
      <td>3.378949</td>
      <td>2.120797</td>
      <td>2084.512372</td>
      <td>1.528082e+04</td>
      <td>1.495147</td>
      <td>0.007613</td>
      <td>0.229984</td>
      <td>3.410862</td>
      <td>7.663748</td>
      <td>1971.111217</td>
      <td>98077.558241</td>
      <td>47.559177</td>
      <td>-122.213520</td>
      <td>12900.415556</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.722258e+05</td>
      <td>0.935301</td>
      <td>0.766772</td>
      <td>918.617686</td>
      <td>4.182288e+04</td>
      <td>0.539352</td>
      <td>0.086924</td>
      <td>0.761324</td>
      <td>0.651961</td>
      <td>1.172238</td>
      <td>29.336823</td>
      <td>53.414906</td>
      <td>0.138629</td>
      <td>0.140706</td>
      <td>27977.230059</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.200000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1900.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>659.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.210000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.048500e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1952.000000</td>
      <td>98033.000000</td>
      <td>47.469200</td>
      <td>-122.328000</td>
      <td>5100.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1920.000000</td>
      <td>7.602000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1975.000000</td>
      <td>98065.000000</td>
      <td>47.571000</td>
      <td>-122.229000</td>
      <td>7620.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.448750e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.072000e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>1997.000000</td>
      <td>98117.000000</td>
      <td>47.677400</td>
      <td>-122.124000</td>
      <td>10107.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>871200.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_init.isna().sum()
```




    price          0
    bedrooms       0
    bathrooms      0
    sqft_living    0
    sqft_lot       0
    floors         0
    waterfront     0
    view           0
    condition      0
    grade          0
    yr_built       0
    zipcode        0
    lat            0
    long           0
    sqft_lot15     0
    dtype: int64




```python
# remove outliers and data will be cleaned, can move on to feature engineering
# we can standartize the data as well to do this
# we can see that there are no homes below 3 standard deviations with the describe table
# however there are homes above 3 standard deviations of the mean price
z = abs(stats.zscore(df_init)) 
print(z)
threshold = 3
print(np.where(z>3))
avg_price = df_init['price'].mean()
stdev_price = df_init['price'].std()
upper_bound = avg_price + 3*stdev_price

# df_clean = df_init[(z < 3).all(axis=1)] # this doesnt work because we remove some data (waterfront) that we don't want to remove - remove only homes sold with a price above 3 standard deviations
df_clean = df_init[df_init['price']<upper_bound]
```

    [[0.00891201 0.40517583 0.16850812 ... 1.16734842 0.74967289 0.1880666 ]
     [0.16840532 0.66403252 1.14666608 ... 0.2768393  1.27560722 0.2823963 ]
     [0.08413755 0.40517583 0.15754454 ... 0.41567829 1.19770555 0.19292785]
     ...
     [0.37966644 0.66403252 0.49456077 ... 0.34969792 1.05528338 0.20375843]
     [0.37402184 1.47438418 1.78780781 ... 0.25409085 0.60752848 0.38937955]
     [0.58116341 1.47438418 1.78780781 ... 0.25192673 0.60752848 0.41261346]]
    (array([    3,     3,     3, ..., 15740, 15740, 15751], dtype=int64), array([ 2,  3, 14, ...,  6,  7,  3], dtype=int64))
    


```python
df_clean.isna().sum()
```




    price          0
    bedrooms       0
    bathrooms      0
    sqft_living    0
    sqft_lot       0
    floors         0
    waterfront     0
    view           0
    condition      0
    grade          0
    yr_built       0
    zipcode        0
    lat            0
    long           0
    sqft_lot15     0
    dtype: int64




```python
df_clean.price.max() # have removed absurdly high priced homes from df
df_clean.waterfront.max() # where we discover we removed data that is binary with low standard deviation compared to the max, which is 1. 
# lets try a different method
df_clean.describe()
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
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.548000e+04</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>1.548000e+04</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.084310e+05</td>
      <td>3.362532</td>
      <td>2.093169</td>
      <td>2037.260401</td>
      <td>1.507320e+04</td>
      <td>1.487791</td>
      <td>0.004522</td>
      <td>0.203424</td>
      <td>3.409044</td>
      <td>7.613114</td>
      <td>1971.053682</td>
      <td>98077.850452</td>
      <td>47.558118</td>
      <td>-122.213209</td>
      <td>12791.542442</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.615181e+05</td>
      <td>0.926927</td>
      <td>0.733848</td>
      <td>834.955963</td>
      <td>4.123850e+04</td>
      <td>0.537843</td>
      <td>0.067096</td>
      <td>0.707149</td>
      <td>0.650400</td>
      <td>1.109891</td>
      <td>29.239459</td>
      <td>53.309799</td>
      <td>0.139496</td>
      <td>0.141449</td>
      <td>27915.589542</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.200000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1900.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>659.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.200000e+05</td>
      <td>3.000000</td>
      <td>1.500000</td>
      <td>1420.000000</td>
      <td>5.012750e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1952.000000</td>
      <td>98033.000000</td>
      <td>47.465475</td>
      <td>-122.329000</td>
      <td>5100.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.490000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1900.000000</td>
      <td>7.560000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1975.000000</td>
      <td>98065.000000</td>
      <td>47.569150</td>
      <td>-122.229000</td>
      <td>7584.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.275000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2510.000000</td>
      <td>1.050000e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>1996.000000</td>
      <td>98117.000000</td>
      <td>47.678200</td>
      <td>-122.123000</td>
      <td>10014.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.650000e+06</td>
      <td>33.000000</td>
      <td>7.500000</td>
      <td>7350.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>12.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>871200.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# after having completed a regression, discovered that lat and long are not worthwhile in a regression analysis
# as a result, will create a distance from seattle feature to create a regression 
```


```python
df_clean['lat'] = [round(i,4) for i in df_clean['lat']]
df_clean['long'] = [round(i,4) for i in df_clean['long']]
```


```python
df_clean['geo_loc'] = list(zip(df_clean['lat'], df_clean['long']))
```


```python
Seattle = [47.6219, -122.3517] # defining Seattle's location
```


```python
distance = []
for i in df_clean['geo_loc']:
    distance.append((haversine((Seattle),(i), unit='mi')))
rounded = [round(i,4) for i in distance]

df_clean['distance'] = rounded
```


```python
df_clean.info()
df_clean.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 15480 entries, 1 to 21596
    Data columns (total 17 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   price        15480 non-null  float64
     1   bedrooms     15480 non-null  int64  
     2   bathrooms    15480 non-null  float64
     3   sqft_living  15480 non-null  int64  
     4   sqft_lot     15480 non-null  int64  
     5   floors       15480 non-null  float64
     6   waterfront   15480 non-null  float64
     7   view         15480 non-null  float64
     8   condition    15480 non-null  int64  
     9   grade        15480 non-null  int64  
     10  yr_built     15480 non-null  int64  
     11  zipcode      15480 non-null  int64  
     12  lat          15480 non-null  float64
     13  long         15480 non-null  float64
     14  sqft_lot15   15480 non-null  int64  
     15  geo_loc      15480 non-null  object 
     16  distance     15480 non-null  float64
    dtypes: float64(8), int64(8), object(1)
    memory usage: 2.8+ MB
    




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
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_lot15</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.548000e+04</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>1.548000e+04</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
      <td>15480.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.084310e+05</td>
      <td>3.362532</td>
      <td>2.093169</td>
      <td>2037.260401</td>
      <td>1.507320e+04</td>
      <td>1.487791</td>
      <td>0.004522</td>
      <td>0.203424</td>
      <td>3.409044</td>
      <td>7.613114</td>
      <td>1971.053682</td>
      <td>98077.850452</td>
      <td>47.558118</td>
      <td>-122.213209</td>
      <td>12791.542442</td>
      <td>12.126486</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.615181e+05</td>
      <td>0.926927</td>
      <td>0.733848</td>
      <td>834.955963</td>
      <td>4.123850e+04</td>
      <td>0.537843</td>
      <td>0.067096</td>
      <td>0.707149</td>
      <td>0.650400</td>
      <td>1.109891</td>
      <td>29.239459</td>
      <td>53.309799</td>
      <td>0.139498</td>
      <td>0.141449</td>
      <td>27915.589542</td>
      <td>7.098748</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.200000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1900.000000</td>
      <td>98001.000000</td>
      <td>47.156000</td>
      <td>-122.519000</td>
      <td>659.000000</td>
      <td>0.357500</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.200000e+05</td>
      <td>3.000000</td>
      <td>1.500000</td>
      <td>1420.000000</td>
      <td>5.012750e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1952.000000</td>
      <td>98033.000000</td>
      <td>47.465000</td>
      <td>-122.329000</td>
      <td>5100.000000</td>
      <td>6.281050</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.490000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1900.000000</td>
      <td>7.560000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1975.000000</td>
      <td>98065.000000</td>
      <td>47.569000</td>
      <td>-122.229000</td>
      <td>7584.000000</td>
      <td>10.776850</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.275000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2510.000000</td>
      <td>1.050000e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>1996.000000</td>
      <td>98117.000000</td>
      <td>47.678000</td>
      <td>-122.123000</td>
      <td>10014.500000</td>
      <td>16.846850</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.650000e+06</td>
      <td>33.000000</td>
      <td>7.500000</td>
      <td>7350.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>12.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.778000</td>
      <td>-121.315000</td>
      <td>871200.000000</td>
      <td>48.629200</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_clean.isna().sum()
```




    price          0
    bedrooms       0
    bathrooms      0
    sqft_living    0
    sqft_lot       0
    floors         0
    waterfront     0
    view           0
    condition      0
    grade          0
    yr_built       0
    zipcode        0
    lat            0
    long           0
    sqft_lot15     0
    geo_loc        0
    distance       0
    dtype: int64




```python
# send data to a csv!
df_clean.to_csv(r'data\CleanHousing.csv')

```


```python
# checking for OLS assumptions:
# 1) the regression model is 'linear in parameters'
# 2) There is a random sampling of observations
# 3) The conditional mean should be zero, ie the expected value of the mean of the error terms should be zero
# 4) There is no multi-collinearity (no features can be derived from other features' values)
# 5) There is homoscedasticity and no autocorrelation
# 6) Error terms should be normally distributed
```


```python

```
