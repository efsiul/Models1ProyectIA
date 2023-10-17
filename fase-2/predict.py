import  argparse
import  os
import  pandas as pd
import  pickle
from    loguru import logger

DEFAULT_INPUT_FILE          = 'data/uber_test.csv'
DEFAULT_PREDICTIONS_FILE    = 'data/predictions.csv'
DEFAULT_MODEL_FILE          = 'models/model.pkl'
DEFAULT_PREPROCESSOR_FILE   = 'models/preprocessor.pkl'

parser                      = argparse.ArgumentParser()
parser.add_argument('--input_file',         default=DEFAULT_INPUT_FILE,        type=str, help='a csv file with input data (no targets)')
parser.add_argument('--predictions_file',   default=DEFAULT_PREDICTIONS_FILE,  type=str, help='a csv file where predictions will be saved to')
parser.add_argument('--model_file',         default=DEFAULT_MODEL_FILE,        type=str, help='a .pkl file with a model already stored (see train.py)')
parser.add_argument('--preprocessor_file',  default=DEFAULT_PREPROCESSOR_FILE, type=str, help='a .pkl file with a preprocessor already stored (see train.py)')


# Analizar los argumentos de la línea de comandos
args = parser.parse_args()

# Obtener los valores de los argumentos para su posterior uso
model_file          = args.model_file
preprocessor_file   = args.preprocessor_file
input_file          = args.input_file
predictions_file    = args.predictions_file

# Comprobar si el archivo del modelo especificado existe
if not os.path.isfile(preprocessor_file):
    logger.error(f"preprocessor file {preprocessor_file} does not exist")
    exit(-1)


if not os.path.isfile(model_file):
    logger.error(f"model file {model_file} does not exist")
    exit(-1)


# Comprobar si el archivo de entrada especificado existe
if not os.path.isfile(input_file):
    logger.error(f"input file {input_file} does not exist")
    exit(-1) 


logger.info("loading input data")
Xts = pd.read_csv(input_file).iloc[:,:]


# Cargar el preprocesador y el modelo previamente entrenado
preprocessor    = pickle.load(open(preprocessor_file, 'rb'))
model           = pickle.load(open(model_file, 'rb'))


logger.info("making predictions")
X_input         = preprocessor.transform(Xts)                                   # Realizar el preprocesamiento de los datos de entrada
predictions     = model.predict(X_input)                                        # Realizar las predicciones



# Imprimir las predicciones o guardarlas según tus necesidades
for prediction in predictions:
    print(f'Predicción: {prediction}')
    
logger.info(f"saving predictions to {predictions_file}")
pd.DataFrame(predictions.reshape(-1,1), columns=['preds']).to_csv(predictions_file, index=False)

