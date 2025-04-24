import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump, load
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
# The transform method applies the learned parameters (obtained during the fit) to the dataset.
ytrain = pd.DataFrame(min_max_scaler_y.transform(ytrain), columns = ytrain.columns)


min_max_scaler_X.fit(xtest)  # Fit learn the relevant parameters before applying those parameters to transform your data.
min_max_scaler_y.fit(ytest)
xtest = pd.DataFrame(min_max_scaler_X.transform(xtest), columns = xtest.columns)
ytest = pd.DataFrame(min_max_scaler_y.transform(ytest), columns = ytest.columns)

# LR_model = LinearRegression()
# LR_model.fit(xtrain, ytrain)

# # dump(min_max_scaler_X,'scaler.pkl')
# dump(LR_model, 'LR_model.pkl')
lr_model = load('LR_model.pkl')

r2_lr_train = lr_model.score(xtrain, ytrain)
print("Linear Regression train accuracy",round(r2_lr_train*100,2), "%")
r2_lr_test = lr_model.score(xtest, ytest)
print("Linear Regression test accuracy",round(r2_lr_test*100,2), "%")

