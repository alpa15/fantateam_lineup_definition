# LIBRARIES
import numpy as np
import pandas as pd
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from sklearn.metrics import mean_squared_error

## CLASS: A neural network model is created thanks to this class and
##        it is composed of 2 functions, the one for training and the
##        one for predicting the outputs
class NeuralNetwork():
    ## GOAL: Class initialization
    def __init__(self):
        self.model = Sequential()

    ## GOAL: Starting from the time series, it is normalized (or denormalized)
    def normalize(self, normalization_type, arr, params_for_norm):
        arr_norm = arr.copy()
        arr_norm = np.nan_to_num(arr_norm)
        params = []
        match normalization_type:
            case 'min-max':
                if params_for_norm != '':
                    arr_norm = (arr - params_for_norm[0]) / (
                    params_for_norm[1] - params_for_norm[0])
                else:
                    arr_norm = (arr - np.min(arr, axis=0)) / (
                        np.max(arr, axis=0) - np.min(arr, axis=0))
                    params = [np.min(arr, axis=0), np.max(arr, axis=0)]
            case 'z-score':
                if params_for_norm != '':
                    arr_norm = (arr - params_for_norm[0]) / params_for_norm[1]
                else:
                    std = np.array([x if np.abs(x) >= np.finfo(np.float32).eps else 1
                            for x in np.std(arr, axis=0)])
                    arr_norm = (arr - np.mean(arr, axis=0)) / std
                    params = [np.mean(arr, axis=0), std]
            case 'l1':
                if params_for_norm != '':
                    arr_norm = arr / params_for_norm[0]
                else:
                    arr_norm = arr / np.sum(np.abs(arr), axis=0)
                    params = [np.sum(np.abs(arr), axis=0)]
            case 'l2':
                if params_for_norm != '':
                    arr_norm = arr / params_for_norm[0]
                else:
                    arr_norm = arr / np.sqrt(np.sum(arr**2, axis=0))
                    params = [np.sqrt(np.sum(arr**2, axis=0))]
            case 'log':
                arr_norm = np.log(arr)
            case 'sigmoid':
                arr_norm = 1 / (1 + np.exp(-arr))
            case 'no-norm':
                arr_norm = arr
        return arr_norm, params

    ## GOAL: 
    def train(self, data, input_variables, output_variables, params, save_out = False):
        self.model.add(Dense(4, input_dim=2, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=['accuracy'])
        # Normalize data
        input_norm, param_in = self.normalize(params['norm_type'],
                                data[input_variables], params['params_for_norm'])
        output_norm, param_out = self.normalize(params['norm_type'],
                                data[output_variables], params['params_for_norm'])
        # Train the model
        X_train = input_norm
        y_train = output_norm
        self.model.fit(X_train, y_train, epochs=1000, verbose=0)
        if save_out:
            self.model.save('Fantateam lineup definition//' +
                                'nn_model.bin')
    
    ## GOAL: 
    def predict(self, data, input_variables):
        X = np.array(data[[input_variables]])
        y_pred = self.model.predict(X)
        return y_pred


## CLASS: A xgboost model is created thanks to this class and
##        it is composed of 2 functions, the one for training and the
##        one for predicting the outputs
class XgBoost():
    ## GOAL: Class inizialization
    def __init__(self) -> None:
        pass

    # GOAL: Given the training data, the model can be trained thanks to this
    #       function, with the goal of finding the best Xgboost
    def train(self, data, input_variables, output_variables, params, save_out = False):
        # Split the data into training and testing sets
        train = data.sample(frac=0.8, random_state=1)
        test = data.drop(train.index)
        # Create the XGBoost DMatrix
        dtrain = xgb.DMatrix(train[input_variables], label=train[output_variables])
        dtest = xgb.DMatrix(test[input_variables], label=test[output_variables])
        # Train the XGBoost model
        evals_result = {}
        model = xgb.train(params, dtrain, num_boost_round=100,
                        evals=[(dtest, 'test')], evals_result = evals_result)
        print(evals_result)
        if save_out:
            model.save_model('Fantateam lineup definition//' +
                            'xgb_model.bin')
            
    # GOAL: Starting from the model already trained, the data which are passed
    #       are the inputs that gives the possibility to calculate the prediction
    #       of fantavote of every player
    def predict(self, test_df, input_variables):
        test_df[input_variables] = test_df[input_variables
                            ].apply(pd.to_numeric, errors='coerce')
        dtest = xgb.DMatrix(test_df[input_variables])
        # Make predictions using the loaded model
        predictions = self.predict(dtest)
        return predictions