from sklearn.preprocessing import MinMaxScaler


def minmax_normalize(X, scaler=None):
    """
    If scaler is None: fit a new MinMaxScaler.
    Else: transform using an existing scaler (fit on train, reused on test).
    """
    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler
