from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_cancellation_model(df):

    X = df[['distance', 'hour', 'rush_hour']]
    y = df['ride_status']  # Completed / Cancelled / Incomplete

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    pickle.dump(model, open("models/cancellation_model.pkl", "wb"))

    return model