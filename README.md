# 13_Neural-Network

### This project aims to create a model that predicts whether businesses will become successful based on a variety of information give about each business. This information will be used as features to create a binary classifier model using a deep neural network that will predict whether the business applicant will become a successful if funding is received.

---

## Technologies

This project leverages python 3.9 and [Google Colab](https://colab.research.google.com/?utm_source=scs-index) was used to run all analysis.

---

## Installations

Before running the application first install and import the following libraries and dependencies.

```python
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
```

---

## Data Preparation

The dataset contained both categorical and numerical variables. In order to analyze them, we encoded the categorical variables so they are transformed into numerical values, specifically into binary classification.

`OneHotEncoder` was imported and it was used to numerically encode all of the dataset's categorical data, where then the new dataset was saved into a new DataFrame. Below is the code that creates the `OneHotEncoder` instance:

```python
enc = OneHotEncoder(sparse=False)
```

Next, features (x) and target (y) were created. The target was set to `IS_SUCCESSFUL` column and the features were set to all other columns. `StandardScaler` was used to scale the split data.

---

## Creating a Neural Network Model

Below are the codes used for creating a deep neural network where the number of input features, layers, and neurons on each layer were assigned when using [Tensorflow's Keras](https://www.tensorflow.org/api_docs/python/tf/keras).

```python
number_input_features = 116
hidden_nodes_layer1 = 58

nn = Sequential()

nn.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

nn.add(Dense(units=hidden_nodes_layer2, activation="relu"))

nn.add(Dense(units=1, activation="sigmoid"))
```

Then, we compile and then fit our deep neural network model. The following code compiles and fits the model:

```python
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

fit_model = nn.fit(X_train_scaled, y_train, epochs=50)
```

Finally, the model was evaluated and saved to an HDF5 file.

---

## Optimizing the Neural Network Model

Two more alternative models were built in order to improve on the first model's predictive accuracy. Below is the summary of the results:

- **Original Model**:
  268/268 - 0s - loss: 0.5574 - accuracy: 0.7300 - 286ms/epoch - 1ms/step
  Loss: 0.5574018359184265, Accuracy: 0.7300291657447815

* **Alternative Model 1**: `epochs=100` was used
  268/268 - 0s - loss: 0.5547 - accuracy: 0.7283 - 371ms/epoch - 1ms/step
  Loss: 0.5547268986701965, Accuracy: 0.7282798886299133

- **Alternative Model 2**: a second hidden layer was added, `epochs=100`, `activation="leaky_relu"` and `batch_size=32` were used instead
  268/268 - 0s - loss: 0.5606 - accuracy: 0.7297 - 372ms/epoch - 1ms/step
  Loss: 0.5605902671813965, Accuracy: 0.72967928647995

**Conclusion**: Model 2 had the highest accuracy score and therefore, this model would be recommended for use in predicting a business' success.


# Venture Funding with Deep Learning

Alphabet Soup venture capital needs assistance to allocate funds. Its business team receives many funding applications from startups every day. This team needs to create a model that predicts whether applicants will be successful if Alphabet Soup funds them.

The business team provided data from more than 34,000 organizations that have received funding from Alphabet Soup over the years. Using machine learning and neural networks, we create a binary classifier model that will predict whether an applicant will become a successful business. The data contains various information about these businesses, including whether or not they ultimately became successful.


# Parameters Used

The below were the parameters used for the different models.

Model 1:
* Input Features - 116
* Output Layer - 1
* Hidden Layer - 2
* Nodes For Hidden Layer 1 - 58
* Nodes For Hidden Layer 2 - 29

<br>

Model 2:
* Input Features - 116
* Output Layer - 1
* Hidden Layer - 1
* Nodes For Hidden Layer 1 - 78

<br>

Model 3:
* Input Features - 116
* Output Layer - 1
* Hidden Layer - 1
* Nodes For Hidden Layer 1 - 117

## Technologies

The project uses the following technologies:

* `Pandas,` `NumPy` for general programming in Jupyter Lab

* `TensorFlow` and `Keras` for the construction and evaluation of the Deep Learning models. Particularly *Sequential* for the fitting, compilation, and evaluation, and *Dense* for the layer construction. We use a version above 2.0.0.

* `SKLearn` for the preprocessing of the data, particularly `StandardScaler` and `MinMaxScaler` for standardization and normalization; `OneHotEncoder` for treatment of the categorical variables, and `train_test_split` for the separation of the sample in a set to train and a set to validate the model.


## Installation Guide
The file is a jupyter notebook. If you don't have jupyter lab, you can install it following the instruction here:

https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html


If you don't have installed TensorFlow, you can run the following command

`pip install --upgrade tensorflow`




### Usage

The main file is `venture_funding_with_deep_learning.ipynb`, which contains the original model, the results, and the conclusions. 

Results and comparisons are displayed in the main file as well.

These are jupyter notebooks with a pre-run code. You can go through it and see code as well as results. 

If you want to reuse the code and do not have experience in jupyter lab, please refer:
https://www.dataquest.io/blog/jupyter-notebook-tutorial/


# Results

The results after running each models.

Model 1:
* `268/268 - 0s - loss: 0.5530 - accuracy: 0.7301 - 316ms/epoch - 1ms/step`
* `Loss: 0.5530197024345398, Accuracy: 0.7301457524299622`




<br>

Model 2:
* `268/268 - 0s - loss: 0.5584 - accuracy: 0.7251 - 423ms/epoch - 2ms/step`
* `Loss: 0.5583803653717041, Accuracy: 0.7251312136650085`


<br>

Model 3:
* `268/268 - 0s - loss: 0.6028 - accuracy: 0.7308 - 353ms/epoch - 1ms/step`
* `Loss: 0.6028183102607727, Accuracy: 0.7308454513549805`

**Conclusion**: Model 3 had the highest accuracy score and therefore, this model would be recommended for use in predicting a business' success.

## Contributors
This project was coded by Kelvin Le
Contact email: KelvinLe@live.com
LinkedIn profile: www.linkedin.com/in/KelvinLe469


## License
This project uses an MIT license. This license allows you to use the licensed material at your discretion, as long as the original copyright and license are included in your work files. This license does not contain a patent grant and liberates the authors of any liability from using this code.