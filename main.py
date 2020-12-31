# This is a sample Python script.
import pandas as pd
import datetime
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import PredictionModel as predictm



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
def cleandata():
    df = pd.read_csv('taxifare.csv')
    df.head(5)
    # print(df.head(5))
    # print(df.shape)

    # this define the basic definition of the dataset like columns and data type and count
    # print(df.info())

    # Validate the pick up dat eof the taxi
    # print(df['pickup_datetime'])

    # change the datetime to global timings
    # print(pd.to_datetime(df['pickup_datetime'])-datetime.timedelta(hours=4))

    # Load the standard time in the data frame
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime']) - datetime.timedelta(hours=4)
    df['Year'] = df['pickup_datetime'].dt.year
    df['Month'] = df['pickup_datetime'].dt.month
    df['Day'] = df['pickup_datetime'].dt.day
    df['Hours'] = df['pickup_datetime'].dt.hour
    df['Minutes'] = df['pickup_datetime'].dt.minute
    # print(df['Minutes'])

    # So if you use the below command it will show the increased column that has been added above
    # print(df.info())

    # Taxi has different peak time- Morning and night 0 for Morning and 1 for  evening
    df['Morn/Eve'] = np.where(df['Hours'] < 12, 0, 1)
    # print(df['Morn/Eve'])

    # drop the unnecessary info
    df.drop('pickup_datetime', axis=1)

    # Calculate the distance between the cities, for the Haver function can be used
    # sample code to calculate the distance
    bsas = [-34.83333, -58.5166646]
    paris = [49.0083899664, 2.53844117956]
    bsas_in_radians = [radians(_) for _ in bsas]
    paris_in_radians = [radians(_) for _ in paris]
    result = haversine_distances([bsas_in_radians, paris_in_radians])
    result * 6371000 / 1000  # multiply by Earth radius to get kilometers
    print(result)

    # df['StartingPoint']= [radians(_) for _ in df['']]

    # calculate the distance in a new data frame, this is best value to get the values
    # x = pd.DataFrame(df, columns=["pickup_longitude", "pickup_latitude"])
    # print(x)

    ###https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points

    def haversine(df):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        lat1 = np.radians(df["pickup_latitude"])
        lat2 = np.radians(df["dropoff_latitude"])
        #### Based on the formula  x1=drop_lat,x2=dropoff_long
        dlat = np.radians(df['dropoff_latitude'] - df["pickup_latitude"])
        dlong = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2) ** 2

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        r = 6371  # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    # Calculate the distance

    df['Total distance'] = haversine(df)

    # print(df['Total distance'])
    # print(df)

    # drop off some of the not necessary featiures
    df.drop(["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"], axis=1, inplace=True)
    # print(df.head(5))
    # to make it drop permanently you have to define th INPLACE =TRUE

    # Save the featured csv data into a a new final csv file
    df.to_csv("Final_data.csv")
    #######################################################################
    # Applying the regression model on the problem
    #######################################################################


if __name__ == '__main__':
    print_hi('PyCharm')
    # Prepare the clean and feature the date
    cleandata()
    predictm.predictmodel()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
















