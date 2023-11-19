# Utiliza la imagen oficial de Python como base
FROM python:3.11.6

# Variables de entorno para evitar la salida de texto descriptivo
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Crear y configurar el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar archivos y directorios necesarios al contenedor
COPY data/              /app/data/
COPY models/            /app/models/
COPY apirest.py         /app
COPY requirements.txt   /app

# Instalar las dependencias especificadas en requirements.txt
RUN pip install -r requirements.txt

# Expone los puertos si es necesario
EXPOSE 9000


# Comando para ejecutar la aplicación
CMD ["uvicorn", "apirest:app", "--host", "0.0.0.0", "--port", "9000"]


