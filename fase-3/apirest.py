import  pickle
import  pandas                  as pd
from    fastapi                 import FastAPI, HTTPException, UploadFile, File
from    fastapi.responses       import JSONResponse
from    loguru                  import logger
from    sklearn.compose         import ColumnTransformer
from    sklearn.pipeline        import Pipeline
from    sklearn.preprocessing   import StandardScaler
from    sklearn.ensemble        import GradientBoostingRegressor
from    sklearn.model_selection import train_test_split

app = FastAPI()

DEFAULT_MODEL_FILE              = 'models/model.pkl'
DEFAULT_PREPROCESSOR_FILE       = 'models/preprocessor.pkl'
DEFAULT_DATA_TRAIN_FILE         = 'data/dataTrain.csv'

model                           = pickle.load(open(DEFAULT_MODEL_FILE, 'rb'))
preprocessor                    = pickle.load(open(DEFAULT_PREPROCESSOR_FILE, 'rb'))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
        try:
                df              = pd.read_csv(file.file)                                            # Cargar los datos de entrada desde un archivo CSV      
                X_input         = preprocessor.transform(df)                                        # Realizar el preprocesamiento de los datos de entrada
                predictions     = model.predict(X_input)                                            # Realizar las predicciones
                return JSONResponse(content={"predictions": predictions.tolist()}, status_code=200) # Devolver las predicciones en formato JSON

        except Exception as e:
                logger.error(f"Error en el endpoint predict: {str(e)}")
                raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.post("/train")
async def train(file: UploadFile = File(...)):
        try:
                df_train = pd.read_csv(file.file)                                                   # Leer el archivo CSV de entrenamiento
                df_train.to_csv(DEFAULT_DATA_TRAIN_FILE, index=False)                               # Guardar el archivo de entrenamiento en data/dataTrain.csv


                df_uber_rel               = pd.read_csv(DEFAULT_DATA_TRAIN_FILE)

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
                with open(DEFAULT_PREPROCESSOR_FILE, 'wb') as preprocessor_file:
                        pickle.dump(preprocessor, preprocessor_file)

                with open(DEFAULT_MODEL_FILE, 'wb') as model_file:
                        pickle.dump(grad_reg, model_file)

        except Exception as e:
                logger.error(f"Error en el endpoint train: {str(e)}")
                raise HTTPException(status_code=500, detail="Error interno del servidor")
