"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import Lasso

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
X_train = train[['Madrid_wind_speed', 'Madrid_rain_1h', 'Madrid_pressure', 'Madrid_temp', 'Madrid_humidity']]

# Fit model
lasso = Lasso(alpha=1.5, normalize=True, random_state=42)
print ("Training Model...")
lasso.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/lasso.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lasso, open(save_path,'wb'))
