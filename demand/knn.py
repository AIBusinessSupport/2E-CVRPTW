import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

#from ECVRP import TwoECVrp

#Get the Distance between Customers
def KNN(bound_demand, coord_cus):
   
    ## Get the demands of customers, the shape is (n_customers, 3)
    mean_demand = bound_demand[:,1]
    coord_customers = coord_cus    ## Get the Coordinations of customers, the shape is (n_customers, 2)
    
    ## Get the prediction of demand of each customers. i-th value is the predicted demands of i-th customers
    predict_demand = np.zeros(len(bound_demand))
    for i in range(len(bound_demand)):
        # Process Training data X and labels y
        X = coord_customers.copy()
        y = mean_demand - np.min(mean_demand)
        y = y.tolist()
        
        # Remove i-th value
        X.remove(X[i])
        y.remove(y[i])
        
        # Normalize the data
        scaler = MinMaxScaler()
        X1 = scaler.fit_transform(X)
        
        # Build the K-NN Regression Model and training
        n_neighbors = int(np.sqrt(len(bound_demand))) + 1
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        model.fit(X1, y)
        
        # Get the prediction
        X_pred = []
        X_pred.append(coord_customers[i])
        # Normalization
        X_pred = scaler.transform(X_pred)
        y_pred = model.predict(X_pred)[0] + np.min(mean_demand)
        
        # Get the real demand:
        predict_demand[i] = min(bound_demand[i, 2], max(bound_demand[i, 0], y_pred))
        
    return predict_demand     
        
    
    