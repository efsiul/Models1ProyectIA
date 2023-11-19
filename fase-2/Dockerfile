# Utiliza la imagen oficial de Python como base
FROM python:3.11.6

# Variables de entorno para evitar la salida de texto descriptivo
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Crear y configurar el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar archivos y directorios necesarios al contenedor
COPY models/            /app
COPY train.py           /app
COPY predict.py         /app
COPY data/              /app
COPY requirements.txt   /app

# Instalar las dependencias especificadas en requirements.txt
RUN pip install -r requirements.txt

# Expone los puertos si es necesario
EXPOSE 8080

# Comando para ejecutar el script predict.py (ajusta el nombre del script y los argumentos según sea necesario)

# CMD ["python", "predict.py","--input_file", "data/uber_test.csv", "--predictions_file", "predictions.csv"]


# Después de ejecutar train.py, ejecuta predict.py
CMD ["python", "train.py"] and ["python",  "predict.py"]


