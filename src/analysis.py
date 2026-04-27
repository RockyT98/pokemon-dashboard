import pandas as pd

def top_flop(df, stat, n=10):
    # Restituisce i migliori e i peggiori n(10) Pokémon per una statistica.
    top = df.sort_values(by=stat, ascending=False).head(n)
    flop = df.sort_values(by=stat, ascending=True).head(n)
    return top, flop

def analyze_by_type(df, stat):

    df_types = pd.concat([
        df[["type1", stat]].rename(columns={"type1": "type"}),
        df[["type2", stat]].rename(columns={"type2": "type"})
    ])

    df_types = df_types[df_types["type"].notna()]
    df_types = df_types[df_types["type"] != "None"]

    stats = df_types.groupby("type")[stat].agg(["mean", "count"]).reset_index()
    stats["mean"] = stats["mean"].round(2)

    return stats.sort_values("mean", ascending=False)


def best_type_by_stat(stats_df):
    best = stats_df.loc[stats_df["mean"].idxmax()]
    worst = stats_df.loc[stats_df["mean"].idxmin()]
    return best, worst

def generate_ai_insights(df):

    stats = ["attack", "defense", "speed", "hp", "sp_attack", "sp_defense", "Total"]

    insights = {}

    for stat in stats:

        df_types = pd.concat([
            df[["type1", stat]].rename(columns={"type1": "type"}),
            df[["type2", stat]].rename(columns={"type2": "type"})
        ])

        df_types = df_types[df_types["type"].notna()]
        df_types = df_types[df_types["type"] != "None"]

        grouped = df_types.groupby("type")[stat].mean()

        best_type = grouped.idxmax()
        best_value = grouped.max()

        worst_type = grouped.idxmin()
        worst_value = grouped.min()

        insights[stat] = {
            "best_type": best_type,
            "best_value": round(best_value, 2),
            "worst_type": worst_type,
            "worst_value": round(worst_value, 2)
        }

    return insights