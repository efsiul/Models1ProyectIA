import pickle
import pandas         as pd
import numpy          as np
from statsmodels.formula.api  import ols
from geopy.distance           import great_circle
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import train_test_split
from sklearn.compose          import ColumnTransformer
from sklearn.preprocessing    import StandardScaler
from sklearn.ensemble         import GradientBoostingRegressor

kp                          = pd.read_csv('./data/uber.csv')
dframe                      = kp.copy()

df                          = dframe.drop(['Unnamed: 0'],axis=1)
df                          = df.drop(['key'], axis=1)
df                          = df.drop_duplicates()
df                          = df.dropna()

df_fare_delete              = df[df['fare_amount']<= 0]
index_to_delete_fare        = df_fare_delete.index
df.drop(index_to_delete_fare,inplace = True)


index_to_delete_passenger   = df[df['passenger_count'] == df['passenger_count'].max()].index
df.drop(index_to_delete_passenger,inplace = True)

df['passenger_count']       = np.where(df['passenger_count'] == 0,1,df['passenger_count'])
df['passenger_count'].value_counts()


df.loc[:, 'pickup_datetime']= pd.to_datetime(df['pickup_datetime'])


df['year']          = pd.DatetimeIndex(df['pickup_datetime']).year
df['month']         = pd.DatetimeIndex(df['pickup_datetime']).month_name()
df['day']           = pd.DatetimeIndex(df['pickup_datetime']).day
df['week_day']      = pd.DatetimeIndex(df['pickup_datetime']).day_name()
df['pickup_time']   = pd.DatetimeIndex(df['pickup_datetime']).time
df['period']        = (pd.DatetimeIndex(df['pickup_datetime']).hour % 24 + 4) // 4
df['period'].replace({  1: 'Late Night',
                        2: 'Early Morning',
                        3: 'Morning',
                        4: 'Noon',
                        5: 'Evening',
                        6: 'Night'}, inplace = True)
df['Quarter']       = pd.DatetimeIndex(df['pickup_datetime']).quarter

df.drop('pickup_datetime', axis=1, inplace = True)


df_delete           = df[   (df['pickup_latitude']  > 180)  | (df['pickup_latitude']  <- 180) |
                            (df['dropoff_latitude'] > 180)  | (df['dropoff_latitude'] <- 180) |
                            (df['pickup_longitude'] > 90)   | (df['pickup_longitude'] <- 90)  |
                            (df['dropoff_longitude']> 90)  | (df['dropoff_longitude'] <- 90)  ]
df_delete.shape

index_to_delete     = df_delete.index

df.drop(index_to_delete,inplace=True)


def distance_km(x):
    pickup  = (x['pickup_latitude'], x['pickup_longitude'])
    dropoff = (x['dropoff_latitude'], x['dropoff_longitude'])
    return great_circle(pickup, dropoff).km

df['distance_km']   = df.apply(lambda x: distance_km(x), axis=1)
df[df['distance_km'] == 0].shape


df_uber_rel = df[['fare_amount','passenger_count','distance_km','year','month','week_day','period','Quarter']]


df_uber_rel['period'].replace({ 'Early Morning' : 1,
                                'Morning'       : 2,
                                'Noon'          : 3,
                                'Evening'       : 4,
                                'Night'         : 5,
                                'Late Night'    : 6},inplace=True)

df_uber_rel['month'].replace({  'January'       : 1, 
                                'February'      : 2, 
                                'March'         : 3, 
                                'April'         : 4,
                                'May'           : 5,
                                'June'          : 6,
                                'July'          : 7,
                                'August'        : 8,
                                'September'     : 9, 
                                'October'       : 10, 
                                'November'      : 11, 
                                'December'      : 12},inplace=True)

df_uber_rel['week_day'].replace({
                                'Monday'        : 1,
                                'Tuesday'       : 2,
                                'Wednesday'     : 3,
                                'Thursday'      : 4,
                                'Friday'        : 5,
                                'Saturday'      : 6,
                                'Sunday'        : 7},inplace=True)

df_uber_rel['year'].unique()

df_uber_rel['year'].replace({   2009            : 1,
                                2010            : 2,
                                2011            : 3,
                                2012            : 4,
                                2013            : 5,
                                2014            : 6,
                                2015            : 7},inplace=True)

df_uber_rel.drop(df[df['distance_km'] == 0].index, inplace=True)


for feature in ['fare_amount','distance_km']:
    q1                  = df_uber_rel[feature].quantile(0.25)
    q3                  = df_uber_rel[feature].quantile(0.75)
    iqr                 = q3-q1
    upper_whisker       = q3+1.5*iqr
    lower_whisker       = q1-1.5*iqr
    df_uber_rel[feature]=np.where(df_uber_rel[feature] < lower_whisker,lower_whisker,df_uber_rel[feature])
    df_uber_rel[feature]=np.where(df_uber_rel[feature] > upper_whisker,upper_whisker,df_uber_rel[feature])
    



#ANALISIS DE MULTICOLIENALIDAD
f_passenger_count       = 'passenger_count~	distance_km+year+month+week_day+period+Quarter'
m_passenger_count       = ols(formula=f_passenger_count,data=df_uber_rel).fit()
rsq_passenger_count     = m_passenger_count.rsquared
vif_passenger_count     = round(1/(1-rsq_passenger_count),2)

f_distance_travelled    = '	distance_km~passenger_count+year+month+week_day+period+Quarter'
m_distance_travelled    = ols(formula=f_distance_travelled,data=df_uber_rel).fit()
rsq_distance_travelled  = m_distance_travelled.rsquared
vif_distance_travelled  = round(1/(1-rsq_distance_travelled),2)

f_year                  = 'year~passenger_count+	distance_km+month+week_day+period+Quarter'
m_year                  = ols(formula=f_year,data=df_uber_rel).fit()
rsq_year                = m_year.rsquared
vif_year                = round(1/(1-rsq_year),2)

f_month                 = 'month~passenger_count+	distance_km+year+week_day+period+Quarter'
m_month                 = ols(formula=f_month,data=df_uber_rel).fit()
rsq_month               = m_month.rsquared
vif_month               = round(1/(1-rsq_month),2)

f_week_day              = 'week_day~passenger_count+	distance_km+year+month+period+Quarter'
m_week_day              = ols(formula=f_week_day,data=df_uber_rel).fit()
rsq_week_day            = m_week_day.rsquared
vif_week_day            = round(1/(1-rsq_week_day),2)

f_period                = 'period~passenger_count+	distance_km+year+month+week_day+Quarter'
m_period                = ols(formula=f_period,data=df_uber_rel).fit()
rsq_period              = m_period.rsquared
vif_period              = round(1/(1-rsq_period),2)

f_Quarter               = 'Quarter~passenger_count+	distance_km+year+month+week_day+period'
m_Quarter               = ols(formula=f_Quarter,data=df_uber_rel).fit()
rsq_Quarter             = m_Quarter.rsquared
vif_Quarter             = round(1/(1-rsq_Quarter),2)


vif_dic = {
    'feature':
            ['passenger_count', 'distance_km', 'year', 'month','week_day', 'period', 'Quarter'], 
    'VIF':
            [vif_passenger_count,vif_distance_travelled,vif_year,vif_month,vif_week_day,vif_period,vif_Quarter]}
pd.DataFrame(data=vif_dic)

#Dado que las columnas de mes y trimestre tienen una puntuación VIF superior a 5, muestra que hay multicolinealidad 
# presente. Podemos eliminar la columna Quarter.

f_passenger_count       = 'passenger_count~distance_km+year+month+week_day+period'
m_passenger_count       = ols(formula=f_passenger_count,data=df_uber_rel).fit()
rsq_passenger_count     = m_passenger_count.rsquared
vif_passenger_count     = round(1/(1-rsq_passenger_count),2)

f_distance_travelled    = 'distance_km~passenger_count+year+month+week_day+period'
m_distance_travelled    = ols(formula=f_distance_travelled,data=df_uber_rel).fit()
rsq_distance_travelled  = m_distance_travelled.rsquared
vif_distance_travelled  = round(1/(1-rsq_distance_travelled),2)

f_year                  = 'year~passenger_count+distance_km+month+week_day+period'
m_year                  = ols(formula=f_year,data=df_uber_rel).fit()
rsq_year                = m_year.rsquared
vif_year                = round(1/(1-rsq_year),2)

f_month                 = 'month~passenger_count+distance_km+year+week_day+period'
m_month                 = ols(formula=f_month,data=df_uber_rel).fit()
rsq_month               = m_month.rsquared
vif_month               = round(1/(1-rsq_month),2)

f_week_day              = 'week_day~passenger_count+distance_km+year+month+period'
m_week_day              = ols(formula=f_week_day,data=df_uber_rel).fit()
rsq_week_day            = m_week_day.rsquared
vif_week_day            = round(1/(1-rsq_week_day),2)

f_period                = 'period~passenger_count+distance_km+year+month+week_day'
m_period                = ols(formula=f_period,data=df_uber_rel).fit()
rsq_period              = m_period.rsquared
vif_period              = round(1/(1-rsq_period),2)

vif_dic                 = {
    'feature':
        ['passenger_count', 'distance_km', 'year', 'month','week_day', 'period'], 
    'VIF':
        [vif_passenger_count,vif_distance_travelled,vif_year,vif_month,vif_week_day,vif_period]}
pd.DataFrame(data=vif_dic)


#CONSTRUCCIÓN MODELO

# #Generando un dataset para entrenar el modelo dentro del contenedor
df_uber_rel.to_csv('../fase-2/data/uber_train.csv', index=False)


X                       = df_uber_rel.drop('fare_amount', axis = 1)
X                       = X.drop('Quarter', axis = 1)
y                       = df_uber_rel['fare_amount']


#Generando un nuevo dataset de prueba
df_test                 = X.sample(frac=0.1, random_state=42)
df_test.to_csv('../fase-2/data/uber_test.csv', index=False)



#Haciendo transformación numerica
numerical_transformer   = Pipeline(steps = [('scaler', StandardScaler())])

preprocessor            = ColumnTransformer(transformers = [
        ('num', numerical_transformer,['passenger_count', 'distance_km', 'year', 'month', 'week_day', 'period'])
        ])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train_preprocessed    = preprocessor.fit_transform(X_train)
X_test_preprocessed     = preprocessor.transform(X_test)


grad_reg                = GradientBoostingRegressor(n_estimators=200, max_depth=5)

# Fitting the data
grad_reg.fit(X_train_preprocessed, y_train)

# Checking the score
print('Training Score: ', grad_reg.score(X_train_preprocessed, y_train))
print('Testing Score: ' , grad_reg.score(X_test_preprocessed, y_test))


#SAVE MODEL
pickle.dump(preprocessor,   open('../fase-2/models/preprocessor.pkl', 'wb'))
pickle.dump(grad_reg,       open('../fase-2/models/model.pkl', 'wb'))