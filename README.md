# **Uber Fares Prediction**

for

### Luis Felipe Cadavid Ch

## Phase 1

In the phase-1 folder, there are the files that in summary correspond to the selection of the dataset, the data analysis, the testing of different models and the saving of the selected model 'Gradient Boosting Regressor'.

## Phase 2

There is a set of folders and files where we can find,
data: where files with a csv extension are saved, which are used to train the model and make predictions.

model: Folder where the pkl are located, created by the picklet library to use them from the predict.py file and make the predictions from the file found in the data folder.

src: In this folder you will find the files with python code that allow you to train the model and predict.

Dockerfile: File with which a container is built with the instructions to run the project in a new instance.

Requirements.txt: File that contains all the libraries that will be used in the python scripts found in src.

### How to run the container

Open a terminal in the main project directory and run the following commands:

docker build -t model_fare_amount . \
docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data model_fare_amount
