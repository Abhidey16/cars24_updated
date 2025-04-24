import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
from keras import layers
import warnings
warnings.filterwarnings("ignore")


x = pd.read_csv("x.csv")
y = pd.read_csv("y.csv")

xtrain, xtest, ytrain, ytest  = train_test_split(x, y,test_size = 0.3,random_state = 1 ) ## 30% unseen data

from category_encoders import TargetEncoder

target_encoder = TargetEncoder()
target_encoder.fit(xtrain, ytrain)

xtrain = target_encoder.transform(xtrain)
xtest = target_encoder.transform(xtest)

# scaling the data
from sklearn.preprocessing import MinMaxScaler

min_max_scaler_X = MinMaxScaler()
min_max_scaler_y = MinMaxScaler()

min_max_scaler_X.fit(xtrain)  # Fit learn the relevant parameters before applying those parameters to transform your data.
min_max_scaler_y.fit(ytrain)

xtrain = pd.DataFrame(min_max_scaler_X.transform(xtrain), columns = xtrain.columns)
ytrain = pd.DataFrame(min_max_scaler_y.transform(ytrain), columns = ytrain.columns)


# min_max_scaler_X.fit(xtest)  # Fit learn the relevant parameters before applying those parameters to transform your data.
# min_max_scaler_y.fit(ytest)
xtest = pd.DataFrame(min_max_scaler_X.transform(xtest), columns = xtest.columns)
ytest = pd.DataFrame(min_max_scaler_y.transform(ytest), columns = ytest.columns)

# # Build the Neural Network model
# NN_model = keras.Sequential([
#     layers.Dense(128, activation='relu', input_shape=(xtrain.shape[1],)),  # Input layer
#     layers.Dense(64, activation='relu'),                                   # Hidden layer
#     # layers.Dense(32, activation='relu'),                                 # Hidden layer
#     layers.Dense(1)                                                        # Output layer
# ])

# # Compile the model
# NN_model.compile(optimizer='adam', loss='mean_squared_error')

# history = NN_model.fit(xtrain, ytrain,validation_split = 0.2, epochs=10, batch_size=16,verbose=1)

# dump(min_max_scaler_X,'scaler.pkl')
# dump(NN_model, 'NN_model.pkl')
NN_model = load('NN_model.pkl')

y_pred_train = NN_model.predict(xtrain)
r2_nn_train = r2_score(ytrain, y_pred_train)
print("Neural Network train accuracy",round(r2_nn_train*100,2), "%\n")

y_pred = NN_model.predict(xtest)
r2_nn_test = r2_score(ytest, y_pred)
print("Neural Network test accuracy",round(r2_nn_test*100,2), "%")
