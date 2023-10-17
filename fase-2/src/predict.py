import  argparse
import  os
import  pandas as pd
import  pickle
from    loguru import logger

parser              = argparse.ArgumentParser()
parser.add_argument('--input_file',         required=True, type=str, help='a csv file with input data (no targets)')
parser.add_argument('--predictions_file',   required=True, type=str, help='a csv file where predictions will be saved to')
parser.add_argument('--model_file',         required=False, type=str, help='un archivo .pkl con un modelo ya almacenado (ver train.py)')


# Analizar los argumentos de la línea de comandos
args = parser.parse_args()

# Obtener los valores de los argumentos para su posterior uso
model_file          = args.model_file
input_file          = args.input_file
predictions_file    = args.predictions_file

# Comprobar si el archivo del modelo especificado existe
if model_file is None:
    model_file = 'models/model.pkl'  
elif not os.path.isfile(model_file):
    logger.error(f"model file {model_file} does not exist")
    exit(-1)


# Comprobar si el archivo de entrada especificado existe
if input_file is None:
    input_file = 'data/uber_test.csv'
elif not os.path.isfile(input_file):
    logger.error(f"input file {input_file} does not exist")
    exit(-1) 


logger.info("loading input data")
Xts = pd.read_csv(input_file).values[:,:2]

# Cargar el preprocesador y el modelo previamente entrenado
preprocessor    = pickle.load(open('models/preprocessor.pkl', 'rb'))
model           = pickle.load(open('models/model.pkl', 'rb'))


logger.info("making predictions")
X_input         = preprocessor.transform(Xts)                                   # Realizar el preprocesamiento de los datos de entrada
predictions     = model.predict(X_input)                                        # Realizar las predicciones



# Imprimir las predicciones o guardarlas según tus necesidades
for prediction in predictions:
    print(f'Predicción: {prediction}')
    
logger.info(f"saving predictions to {predictions_file}")
pd.DataFrame(predictions.reshape(-1,1), columns=['preds']).to_csv(predictions_file, index=False)