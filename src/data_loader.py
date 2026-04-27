import pandas as pd

def load_data():
    # Carico il dataset. Questa funzione viene chiamata una sola volta
    df = pd.read_csv("data/pokemon.csv")

    # Gestisco i valori mancanti nel secondo tipo per analisi e non fare errori
    df["type2"] = df["type2"].fillna("None")

    # Rinomina colonne per evitare problemi con spazi
    df.rename(columns={
        "height (m)": "height",
        "weight (kg)": "weight"
    }, inplace=True)

    return df