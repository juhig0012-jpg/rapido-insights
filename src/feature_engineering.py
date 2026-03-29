def create_features(df):

    df['hour'] = df['booking_time'].dt.hour

    # Rush Hour
    df['rush_hour'] = df['hour'].apply(
        lambda x: 1 if (7 <= x <= 10 or 17 <= x <= 21) else 0
    )

    # Fare per KM
    df['fare_per_km'] = df['fare'] / df['distance']

    # Long Distance
    df['long_distance'] = df['distance'].apply(
        lambda x: 1 if x > 10 else 0
    )

    return df