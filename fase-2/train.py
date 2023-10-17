import argparse
import pickle
import pandas as pd
from loguru                     import logger
from sklearn.ensemble           import GradientBoostingRegressor
from sklearn.compose            import ColumnTransformer
from sklearn.preprocessing      import StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.model_selection    import train_test_split


# Definir los valores por defecto para los argumentos
DEFAULT_DATA_FILE         = 'data/uber_train.csv'
DEFAULT_PREPROCESSOR_FILE = 'models/preprocessor.pkl'
DEFAULT_MODEL_FILE        = 'models/model.pkl'

parser                    = argparse.ArgumentParser()
parser.add_argument('--data_file',          type=str, default=DEFAULT_DATA_FILE,         help='a csv file with train data')
parser.add_argument('--preprocessor_file',  type=str, default=DEFAULT_PREPROCESSOR_FILE, help='where the preprocessor will be stored')
parser.add_argument('--model_file',         type=str, default=DEFAULT_MODEL_FILE,        help='where the trained model will be stored')

args                      = parser.parse_args()
data_file                 = args.data_file
preprocessor_file         = args.preprocessor_file
model_file                = args.model_file


# Cargar los datos de entrenamiento desde un archivo CSV
df_uber_rel               = pd.read_csv(data_file)

# Construir el modelo y realizar el entrenamiento
X                         = df_uber_rel.drop('fare_amount', axis=1)
X                         = X.drop('Quarter', axis=1)
y                         = df_uber_rel['fare_amount']

df_test                   = X.sample(frac=0.1, random_state=42)
df_test.to_csv('./data/uber_test.csv', index=False)

numerical_transformer     = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor              = ColumnTransformer(transformers=[
        ('num', numerical_transformer, ['passenger_count', 'distance_km', 'year', 'month', 'week_day', 'period'])
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_preprocessed             = preprocessor.fit_transform(X_train)
X_test_preprocessed              = preprocessor.transform(X_test)
grad_reg                         = GradientBoostingRegressor(n_estimators=200, max_depth=5)

# Fitting the data
grad_reg.fit(X_train_preprocessed, y_train)

# Checking the score
print('Training Score: ', grad_reg.score(X_train_preprocessed, y_train))
print('Testing Score: ' , grad_reg.score(X_test_preprocessed, y_test))

# Guardar el modelo entrenado y el preprocesador
with open(preprocessor_file, 'wb') as preprocessor_file:
    pickle.dump(preprocessor, preprocessor_file)

with open(model_file, 'wb') as model_file:
    pickle.dump(grad_reg, model_file)
