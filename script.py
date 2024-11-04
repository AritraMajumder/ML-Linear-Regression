import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sb
#%matplotlib inline
from util import *
import warnings

warnings.filterwarnings("ignore")

matplotlib.rcParams["figure.figsize"] = (20,10)

df = pd.read_csv("Bengaluru_House_Data.csv")
df.groupby('area_type')['area_type'].agg('count')
df2 = df.drop(['area_type','availability','society','balcony'],axis = 1)
df2.isnull().sum()


#take the non float values of total_sqft and return float
df3 = df2.dropna()
df3['bhk'] = df3['size'].apply(lambda x: int(x.split()[0]))
df3 = df3.drop(['size'],axis=1)
df3[~df3['total_sqft'].apply(isfloat)].head(10)


#all possible measures converted to sq feet
df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4[~df4['total_sqft'].apply(isfloat)].head(10)


#Feature engineering: creating new relevant feature
df5 = df4.copy()
df5['price per sqft'] = round(df5['price']*100000/df5['total_sqft'],3)
df5_stats = df5['price per sqft'].describe()
df5.location = df5.location.apply(lambda x: x.strip())


location_stats = df5['location'].value_counts(ascending=False)

#dimensionality reduction
#if a location has less than 10 data points, group it as other location
loc_stats_less = location_stats[location_stats<=10]


df5['location'] = df5['location'].apply(lambda x: 'other' if x in loc_stats_less else x)
df6 = df5[~(df5.total_sqft/df5.bhk<300)]


#removes points where price per sq feet is over 1 std dev away from mean
df7 = remove_pps_outliers(df6)
plot_scatter_chart(df7,"Rajaji Nagar")

#REMOVING 3BHK APTS WHOSE PRICE/SQFT IS LESS THAN MEAN 
#PRICE/SQFT OF 2BHK APT OF SAME LOCATION because sus
df8 = remove_bhk_outliers(df7)

#remove points with unusual number of bathrooms
df9 = df8[df8.bath<df8.bhk+2]

#ppsqft not needed anymore
df10 = df9.drop('price per sqft',axis=1)



#CLEANING DONE

#getting dummies for location
dum = pd.get_dummies(df10.location)

#dummies with other columns
#drop 'other' column as 0 in all other columns means 'other'
df11 = pd.concat([df10,dum.drop('other',axis=1)],axis = 'columns')

#exclude locations
df12 = df11.drop('location',axis = 'columns')


#MODEL FROM HERE
x = df12.drop('price',axis=1)
y = df12.price

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf= LinearRegression()
lr_clf.fit(x_train,y_train)
print('R^2 score: ',lr_clf.score(x_test,y_test))


#crossvalidation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=10)
print('cross validation score: ',cross_val_score(LinearRegression(), x, y, cv=cv))

#testing
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

X=x
print('Predicted Price: ',predict_price('Indira Nagar',1000,3,3))