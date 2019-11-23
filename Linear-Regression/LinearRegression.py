class LinearRegression:
    
    import numpy as np
    
    def __init__(self):       
        
        self.coefficients = []
        

    def fit(self, X, y):
        
        self.rows = X.shape[0]
        
        if len(X) == 1:
            pass
            
        else:
            rows = X.shape[0]
            ones = np.ones(shape=rows).reshape(-1, 1)
            X = np.concatenate((X, ones), 1)
            X_T = X.transpose()
            X_mod = np.linalg.inv(X_T.dot(X))
            self.coefficients = X_mod.dot(X_T).dot(y)
            
     
    def return_coefficients(self):
        
        return self.coefficients
        
    
    def predict(self):
        
        self.y_preds = self.coefficients[3] + self.coefficients[0] * X[:, 0] + self.coefficients[1] * X[:, 1] + self.coefficients[2] * X[:, 2]
        
        return self.y_preds
        
        
    def score(self, metric="rmse"):
        
        """Provides scoring metrics for linear regression.
        
        Options:
        
        "rmse" = Root Mean Squared Error (default option)
        "rmsle" = Root Mean Squared Logarithmic Error
        "r2" = R-squared       
        
        """      
        
        
        if metric=="rmse":
            error = y - self.y_preds
            sum_of_squared_errors = (error ** 2).sum()
            mean_squared_error = sum_of_squared_errors / self.rows
            rmse = np.sqrt(mean_squared_error)
        
            return rmse
        
        if metric=="rmsle":
            rmsle = np.sqrt( ( ( np.log(self.y_preds+1) - np.log(y+1) ) ** 2).mean() )
                       
            return rmsle

        
        if metric=="r2":
            total_sum_of_squares = ((y - y.mean()) ** 2).sum()
            error_sum_of_squares = ((self.y_preds - y.mean()) ** 2).sum()
            r2 = error_sum_of_squares /  total_sum_of_squares
            
            return r2