import argparse
import pickle
import pandas as pd
import os
from loguru                     import logger
from sklearn.ensemble           import GradientBoostingRegressor
from sklearn.compose            import ColumnTransformer
from sklearn.preprocessing      import StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.model_selection    import train_test_split

# Cargar los datos de entrenamiento desde un archivo CSV
df_uber_rel                         = pd.read_csv('data/uber_train.csv')

# Construir el modelo y realizar el entrenamiento
X                                   = df_uber_rel.drop('fare_amount', axis=1)
X                                   = X.drop('Quarter', axis=1)
y                                   = df_uber_rel['fare_amount']

numerical_transformer               = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor                        = ColumnTransformer(transformers=[
        ('num', numerical_transformer, ['passenger_count', 'distance_km', 'year', 'month', 'week_day', 'period'])
    ])

X_train, X_test, y_train, y_test    = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_preprocessed                = preprocessor.fit_transform(X_train)

grad_reg                            = GradientBoostingRegressor(n_estimators=200, max_depth=5)

grad_reg.fit(X_train_preprocessed, y_train)

# Guardar el modelo entrenado y el preprocesador
with open('preprocessor.pkl', 'wb') as preprocessor_file:
    pickle.dump(preprocessor, preprocessor_file)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(grad_reg, model_file)
