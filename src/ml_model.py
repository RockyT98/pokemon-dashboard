#______________________________________________________________________________________
# MACHINE LEARNING MODEL - POKÉMON PREDICTION
#______________________________________________________________________________________

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import pandas as pd


#______________________________________________________________________________________
# TRAIN MODEL
#______________________________________________________________________________________
def train_model(df, target_stat):

    # FEATURES (NON INCLUDIAMO TOTAL → evita data leakage)
    features = [
        "hp",
        "attack",
        "defense",
        "sp_attack",
        "sp_defense",
        "speed"
    ]

    # rimuove la statistica da predire
    features = [f for f in features if f != target_stat]

    # pulizia dati
    df_clean = df.dropna(subset=features + [target_stat])

    X = df_clean[features]
    y = df_clean[target_stat]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    #__________________________________________________________________________________
    # METRICHE MODELLO
    #__________________________________________________________________________________
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return model, features, mae, r2, importance


#______________________________________________________________________________________
# PREDICTION
#______________________________________________________________________________________
def predict_stat(model, input_data):

    return model.predict([input_data])[0]