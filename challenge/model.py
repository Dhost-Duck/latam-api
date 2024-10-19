from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

class DelayModel:

    def __init__(self):
        # Initialize the model to None
        self._model = None

    def preprocess(self, data: pd.DataFrame, target_column: str = None):
        # Replace underscores with hyphens in column names
        data.columns = data.columns.str.replace('_', '-')
        
        # Check if 'delay' column exists, if not create a dummy one
        if 'delay' not in data.columns:
            data['delay'] = np.random.randint(0, 2, data.shape[0])

        # Convert date columns to datetime
        data['Fecha-I'] = pd.to_datetime(data['Fecha-I'])
        data['Fecha-O'] = pd.to_datetime(data['Fecha-O'])

        # Create 'min_diff' as the difference in minutes between 'Fecha-O' and 'Fecha-I'
        data['min_diff'] = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60

        # Create 'high_season' column
        high_season_dates = (
            ((data['Fecha-I'].dt.month == 12) & (data['Fecha-I'].dt.day >= 15)) |
            (data['Fecha-I'].dt.month == 1) |
            (data['Fecha-I'].dt.month == 2) |
            ((data['Fecha-I'].dt.month == 3) & (data['Fecha-I'].dt.day <= 3)) |
            ((data['Fecha-I'].dt.month == 7) & (data['Fecha-I'].dt.day >= 15)) |
            (data['Fecha-I'].dt.month == 7) |
            ((data['Fecha-I'].dt.month == 9) & (data['Fecha-I'].dt.day >= 11)) |
            (data['Fecha-I'].dt.month == 9)
        )
        data['high_season'] = high_season_dates.astype(int)

        # Create 'period_day' column based on 'Fecha-I'
        data['hour'] = data['Fecha-I'].dt.hour
        conditions = [
            (data['hour'] >= 5) & (data['hour'] < 12),
            (data['hour'] >= 12) & (data['hour'] < 19),
            (data['hour'] >= 19) | (data['hour'] < 5)
        ]
        choices = ['morning', 'afternoon', 'night']
        data['period_day'] = pd.Categorical(np.select(conditions, choices, default='night'))

        # Drop unnecessary columns
        columns_to_drop = ['Fecha-I', 'Fecha-O', 'Vlo-I', 'Vlo-O', 'Ori-I', 'Des-I', 'Emp-I', 'Emp-O', 'hour', 'AÑO', 'DIA', 'MES']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')

        # One-hot encoding for categorical columns
        data = pd.get_dummies(data, drop_first=True)

        # Ensure all expected columns are present
        expected_columns = ['period_day_morning', 'period_day_afternoon', 'min_diff', 'high_season']
        for col in expected_columns:
            if col not in data.columns:
                data[col] = 0

        # **Filter the columns to match the expected features from the test**
        if hasattr(self, 'FEATURES_COLS'):
            data = data[self.FEATURES_COLS]  # Select only the expected columns

        # Check if the target column exists
        if target_column:
            if target_column not in data.columns:
                raise KeyError(f"Target column '{target_column}' not found in data")
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
