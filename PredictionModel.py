# Multiple Liner regression on classified data ->
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def predictmodel():
    dataset = pd.read_csv("final_data.csv")
    # print(dataset)
    #print(dataset.info())

    X = dataset[['fare_class', 'passenger_count', 'Morn/Eve', 'Total distance']]
    y = dataset['fare_amount']

    # Execute the following code to divide the data into two models
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #Training the Algorithm
    # And finally, to train the algorithm we execute the same code as before,
    # using the fit() method of the LinearRegression class:

    from sklearn.linear_model import LinearRegression
    regressor= LinearRegression()
    regressor.fit(X_train, y_train)

    # As said earlier, in case of multivariable linear regression,
    # the regression model has to find the most optimal coefficients for all the attributes.
    # To see what coefficients our regression model has chosen, execute the following script:

    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
    #print(coeff_df)

    # Making Predictions
    # To make pre-dictions on the test data, execute the following script:
    y_pred = regressor.predict(X_test)

    # To compare the actual output values for X_test with the predicted values,
    # execute the following script:
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    #print(df)

    # Evaluating the Algorithm
    # The final step is to evaluate the performance of algorithm.
    # We'll do this by finding the values for MAE, MSE and RMSE.
    # Execute the following script:
    from sklearn import metrics
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))










