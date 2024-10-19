from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

class DelayModel:

    def __init__(self):
        # Initialize the model to None
        self._model = None

    def preprocess(self, data: pd.DataFrame, target_column: str = None):
        # Convert date columns to datetime
        data['Fecha-I'] = pd.to_datetime(data['Fecha-I'])
        data['Fecha_O'] = pd.to_datetime(data['Fecha_O'])

        # Create 'min_diff' as the difference in minutes between 'Fecha_O' and 'Fecha_I'
        data['min_diff'] = (data['Fecha_O'] - data['Fecha_I']).dt.total_seconds() / 60

        # Create 'high_season' column
        high_season_dates = (
            ((data['Fecha_I'].dt.month == 12) & (data['Fecha_I'].dt.day >= 15)) |
            (data['Fecha_I'].dt.month == 1) |
            (data['Fecha_I'].dt.month == 2) |
            ((data['Fecha_I'].dt.month == 3) & (data['Fecha_I'].dt.day <= 3)) |
            ((data['Fecha_I'].dt.month == 7) & (data['Fecha_I'].dt.day >= 15)) |
            (data['Fecha_I'].dt.month == 7) |
            ((data['Fecha_I'].dt.month == 9) & (data['Fecha_I'].dt.day >= 11)) |
            (data['Fecha_I'].dt.month == 9)
        )
        data['high_season'] = high_season_dates.astype(int)

        # Create 'period_day' column based on 'Fecha_I'
        data['hour'] = data['Fecha_I'].dt.hour
        conditions = [
            (data['hour'] >= 5) & (data['hour'] < 12),
            (data['hour'] >= 12) & (data['hour'] < 19),
            (data['hour'] >= 19) | (data['hour'] < 5)
        ]
        choices = ['morning', 'afternoon', 'night']
        data['period_day'] = pd.Categorical(np.select(conditions, choices, default='night'))

        # Drop unnecessary columns, ignore errors if they don't exist
        columns_to_drop = ['Fecha_I', 'Fecha_O', 'Vlo_I', 'Vlo_O', 'Ori_I', 'Des_I', 'Emp_I', 'Emp_O', 'hour', 'AÑO', 'DIA', 'MES']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')

        # One-hot encoding for categorical columns (like 'period_day')
        data = pd.get_dummies(data, drop_first=True)

        # Ensure that all the expected columns are present
        expected_columns = ['period_day_morning', 'period_day_afternoon', 'min_diff', 'high_season']
        for col in expected_columns:
            if col not in data.columns:
                data[col] = 0

        if target_column:
            target = data[target_column]
            features = data.drop(columns=[target_column])
            return features, target
        return data




    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # Initialize a simple Logistic Regression model
        self._model = LogisticRegression(class_weight='balanced')
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> list:
        # Ensure that the model has been initialized and fitted
        if self._model is None:
            raise Exception("Model is not fitted yet. Call 'fit' before predicting.")
        
        # Predict the results
        return self._model.predict(features).tolist()
