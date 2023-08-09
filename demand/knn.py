import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

from ECVRP import TwoECVrp

#Get the Distance between Customers
def KNN(path):
    '''
    path: os path of parameter data.
    '''
    # Get the whole info of Graph 
    graph = TwoECVrp(path=path)
    ## Get the useful info
    n_customers = graph.n_customers
    #cus_summary = np.array(graph.cus_sumary)
    
    temp_demand = graph.demand         ## Get the demands of customers, the shape is (n_customers, 3)
    mean_demand = temp_demand[:,1]
    coord_customers = graph.upload_params_from_excel()['coor_custo']    ## Get the Coordinations of customers, the shape is (n_customers, 2)
    
    ## Get the prediction of demand of each customers. i-th value is the predicted demands of i-th customers
    predict_demand = np.zeros(n_customers)
    for i in range(n_customers):
        # Process Training data X and labels y
        X = coord_customers
        y = mean_demand - np.min(mean_demand)
        y = y.tolist()
        
        # Remove i-th value
        X.pop(i)
        y.pop(i)
        
        # Normalize the data
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        
        # Build the K-NN Regression Model and training
        n_neighbors = int(np.sqrt(n_customers)) + 1
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        model.fit(X, y)
        
        # Get the prediction
        X_pred = []
        X_pred.append(coord_customers[i].tolist())
        # Normalization
        X_pred = scaler.transform(X_pred)
        y_pred = model.predict(X_pred)[0] + np.min(mean_demand)
        
        # Get the real demand:
        predict_demand[i] = min(temp_demand[i, 2], max(temp_demand[i, 0], y_pred))
        
    return predict_demand     
        
    
    