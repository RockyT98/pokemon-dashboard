#______________________________________________________________________________________
# POKÉMON DASHBOARD - APP.PY
#______________________________________________________________________________________
import pandas as pd
import streamlit as st
import plotly.express as px

from src.data_loader import load_data
from src.filters import apply_filters, get_pokemon_category
from src.analysis import top_flop, analyze_by_type, best_type_by_stat, generate_ai_insights
from src.ml_model import train_model, predict_stat

#______________________________________________________________________________________
# CONFIG APP
#______________________________________________________________________________________
st.set_page_config(page_title="Pokémon Dashboard", layout="wide")

st.title("Pokémon Dashboard")
st.markdown("Data Analysis + Machine Learning")

#______________________________________________________________________________________
# LOAD DATA
#______________________________________________________________________________________
df = load_data()

df["category"] = df.apply(get_pokemon_category, axis=1)

#______________________________________________________________________________________
# SIDEBAR MENU
#______________________________________________________________________________________
menu = st.sidebar.radio(
    "Menu",
    ["Overview", "Top/Flop", "Search", "Type Analysis", "ML", "AI Insights", "Correlation & Insights"]
)

#______________________________________________________________________________________
# SIDEBAR FILTERS
#______________________________________________________________________________________
include_legendary = st.sidebar.checkbox("Include Legendary", True)
include_mythical = st.sidebar.checkbox("Include Mythical", True)
include_ultrabeast = st.sidebar.checkbox("Include Ultrabeast", True)
include_ordinary = st.sidebar.checkbox("Include Ordinary", True)

generations = sorted(df["generation"].unique())

selected_gen = st.sidebar.multiselect(
    "Generation",
    generations,
    default=generations
)

# APPLY FILTERS
df_filtered = apply_filters(
    df,
    include_legendary,
    include_mythical,
    include_ultrabeast,
    include_ordinary,
    selected_gen
)

#______________________________________________________________________________________
# SAFETY CHECK
#______________________________________________________________________________________
if df_filtered.empty:
    st.warning("Nessun Pokémon trovato con i filtri selezionati.")
    st.stop()

#______________________________________________________________________________________
# STATISTICS LIST
#______________________________________________________________________________________
stats = [
    "attack", "defense", "speed",
    "hp", "sp_attack", "sp_defense",
    "Total", "weight", "height"
]

stats_ml = [
    "attack", "defense", "speed",
    "hp", "sp_attack", "sp_defense"
]

#______________________________________________________________________________________
# OVERVIEW
#______________________________________________________________________________________
if menu == "Overview":

    st.header("Dataset Overview")

    stat = st.selectbox("Seleziona statistica", stats)

    col1, col2, col3 = st.columns(3)

    col1.metric("Totale Pokémon", len(df_filtered))
    col2.metric("Media", round(df_filtered[stat].mean(), 2))

    max_value = df_filtered[stat].max()

    max_pokemon = df_filtered[df_filtered[stat] == max_value][["name", "generation"]].head(2)

    max_info = ", ".join(
        f"{row['name'].title()} (Gen {row['generation']})"
        for _, row in max_pokemon.iterrows()
    )

    col3.metric("Massimo", max_value, delta=max_info)

    fig = px.histogram(df_filtered, x=stat, nbins=30)
    st.plotly_chart(fig, use_container_width=True)

#______________________________________________________________________________________
# TOP / FLOP
#______________________________________________________________________________________
elif menu == "Top/Flop":

    stat = st.selectbox("Statistica", stats)

    top, flop = top_flop(df_filtered, stat)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10")

        fig_top = px.bar(
            top,
            x="name",
            y=stat,
            color=stat,
            hover_data=["generation", "type1", "type2", "category"]
        )

        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        st.subheader("Flop 10")

        fig_flop = px.bar(
            flop,
            x="name",
            y=stat,
            color=stat,
            hover_data=["generation", "type1", "type2", "category"]
        )

        st.plotly_chart(fig_flop, use_container_width=True)

#______________________________________________________________________________________
# SEARCH
#______________________________________________________________________________________
elif menu == "Search":

    st.header("Search Pokémon")

    name = st.text_input("Nome Pokémon")

    if name:

        result = df_filtered[
            df_filtered["name"].str.contains(name, case=False, na=False)
        ]

        if result.empty:
            st.info("Nessun Pokémon trovato.")
        else:
            cols = [
                "name", "type1", "type2",
                "generation", "category",
                "attack", "defense", "speed",
                "hp", "sp_attack", "sp_defense",
                "Total"
            ]

            st.dataframe(result[cols], use_container_width=True)

#______________________________________________________________________________________
# TYPE ANALYSIS
#______________________________________________________________________________________
#______________________________________________________________________________________
# TYPE ANALYSIS
#______________________________________________________________________________________

elif menu == "Type Analysis":

    st.title("Analisi Statistiche per Tipo Pokémon")

    stat = st.selectbox("Statistica", stats)

    tipo_col = st.radio(
        "Analisi tipo",
        ["type1", "type2", "Entrambi"]
    )

    #______________________________________________________________________________________
    # PREPARAZIONE DATI
    #______________________________________________________________________________________

    if tipo_col == "Entrambi":

        df_types = pd.concat([
            df_filtered[['type1', stat]].rename(columns={'type1': 'type'}),
            df_filtered[['type2', stat]].rename(columns={'type2': 'type'})
        ])

        df_types = df_types[
            df_types['type'].notna() & (df_types['type'] != "None")
        ]

    else:

        df_types = df_filtered[[tipo_col, stat]].rename(columns={tipo_col: 'type'})

        if tipo_col == "type2":
            df_types['type'] = df_types['type'].fillna("Puro")
            df_types.loc[df_types['type'] == "None", 'type'] = "Puro"

    #______________________________________________________________________________________
    # AGGREGAZIONE
    #______________________________________________________________________________________

    tipo_stats = df_types.groupby('type')[stat].agg(['mean', 'count']).reset_index()
    tipo_stats['mean'] = tipo_stats['mean'].round(2)
    tipo_stats.sort_values(by='mean', ascending=False, inplace=True)

    #______________________________________________________________________________________
    # GRAFICO MEDIA
    #______________________________________________________________________________________

    st.subheader(f"Media di {stat} per tipo")

    fig_bar = px.bar(
        tipo_stats[tipo_stats['type'] != "Puro"],
        x='type',
        y='mean',
        text='mean',
        color='mean',
        color_continuous_scale=px.colors.sequential.Viridis
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    #______________________________________________________________________________________
    # GRAFICO COUNT
    #______________________________________________________________________________________

    st.subheader("Distribuzione Pokémon per tipo")

    fig_count = px.bar(
        tipo_stats,
        x='type',
        y='count',
        text='count',
        color='count',
        color_continuous_scale=px.colors.sequential.Plasma
    )

    st.plotly_chart(fig_count, use_container_width=True)

    #______________________________________________________________________________________
    # INSIGHT AUTOMATICI
    #______________________________________________________________________________________

    tipo_stats_media = tipo_stats[tipo_stats['type'] != "Puro"]

    max_tipo = tipo_stats_media.loc[tipo_stats_media['mean'].idxmax()]
    min_tipo = tipo_stats_media.loc[tipo_stats_media['mean'].idxmin()]

    st.markdown("---")

    col1, col2 = st.columns(2)

    col1.metric(
        "Tipo migliore",
        max_tipo['type'],
        f"{max_tipo['mean']}"
    )

    col2.metric(
        "Tipo peggiore",
        min_tipo['type'],
        f"{min_tipo['mean']}"
    )

#______________________________________________________________________________________
# ML
#______________________________________________________________________________________
elif menu == "ML":

    st.header("Machine Learning")

    target_stat = st.selectbox("Statistica da predire", stats_ml)

    model, features, mae, r2, importance = train_model(df_filtered, target_stat)

    st.markdown("---")

    st.subheader("Predizione")

    input_data = []

    for f in features:
        value = st.slider(
            f,
            float(df_filtered[f].min()),
            float(df_filtered[f].max()),
            float(df_filtered[f].mean())
        )

        input_data.append(value)

    if st.button("Predici"):
        result = predict_stat(model, input_data)

        st.success(f"Predizione {target_stat}: {result:.2f}")

    st.markdown("---")

    st.subheader("Model Performance")

    st.metric("MAE", round(mae, 2))
    st.metric("R² Score", round(r2, 3))

    st.subheader("Feature Importance")

    fig = px.bar(importance, x="feature", y="importance")

    st.plotly_chart(fig, use_container_width=True)
#______________________________________________________________________________________
# AI INSIGHTS
#______________________________________________________________________________________
elif menu == "AI Insights":

    st.header("AI Insights")

    insights = generate_ai_insights(df_filtered)

    for stat, info in insights.items():

        st.markdown("---")
        st.subheader(stat.upper())

        st.write(f"Best Type: {info['best_type']} ({info['best_value']})")
        st.write(f"Worst Type: {info['worst_type']} ({info['worst_value']})")

#______________________________________________________________________________________
# CORRELATION & INSIGHTS
#______________________________________________________________________________________
elif menu == "Correlation & Insights":

    st.header("Correlation Analysis & AI Insights")

    #__________________________
    # SELEZIONE STATISTICA
    #__________________________
    selected_stat = st.selectbox("Seleziona statistica", ["Tutte"] + stats_ml)

    #__________________________
    # CORRELATION MATRIX
    #__________________________
    st.subheader("Correlation Matrix")

    corr = df_filtered[stats_ml].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Correlation between stats"
    )

    st.plotly_chart(fig, use_container_width=True)

    #__________________________
    # AI INSIGHTS
    #__________________________
    st.subheader("AI Insights")

    insights = []
    threshold = 0.6

    # 1. CORRELAZIONI
    for col in corr.columns:
        for idx in corr.index:

            if col != idx:
                value = corr.loc[idx, col]

                if abs(value) >= threshold:

                    if selected_stat != "Tutte" and col != selected_stat and idx != selected_stat:
                        continue

                    if value > 0:
                        insights.append(
                            f"{col} e {idx} crescono insieme ({round(value,2)})."
                        )
                    else:
                        insights.append(
                            f"{col} cresce mentre {idx} diminuisce ({round(value,2)})."
                        )

    # 2. MIGLIOR PREDITTOR
    for target in stats_ml:

        if selected_stat != "Tutte" and target != selected_stat:
            continue

        correlations = corr[target].drop(target)

        best_feature = correlations.abs().idxmax()
        best_value = correlations[best_feature]

        if best_value > 0:
            insights.append(
                f"{best_feature} è il miglior indicatore per {target}."
            )
        else:
            insights.append(
                f"{best_feature} è inversamente legata a {target}."
            )

    # 3. VARIABILITÀ
    for stat in stats_ml:

        if selected_stat != "Tutte" and stat != selected_stat:
            continue

        std = df_filtered[stat].std()
        mean = df_filtered[stat].mean()

        if std > mean * 0.5:
            insights.append(f"{stat} varia molto tra i Pokémon.")
        else:
            insights.append(f"{stat} è abbastanza stabile.")

    #__________________________
    # OUTPUT
    #__________________________
    insights = list(set(insights))

    if insights:
        for text in insights[:12]:
            st.write(f"- {text}")
    else:
        st.info("Nessun insight rilevante trovato")

    #__________________________
    # AI SUMMARY
    #__________________________
    st.subheader("AI Summary")

    summary = []

    # 1. STATISTICA PIÙ VARIABILE
    variability = {
        stat: df_filtered[stat].std()
        for stat in stats_ml
    }

    most_variable = max(variability, key=variability.get)
    least_variable = min(variability, key=variability.get)

    summary.append(
        f"La statistica più variabile è {most_variable}, mentre la più stabile è {least_variable}."
    )

    # 2. RELAZIONE PRINCIPALE
    corr_values = corr.copy()

    for col in corr_values.columns:
        corr_values.loc[col, col] = 0  # elimina auto-correlazione

    max_corr = corr_values.abs().stack().idxmax()
    val = corr.loc[max_corr]

    if val > 0:
        summary.append(
            f"La relazione più forte è tra {max_corr[0]} e {max_corr[1]}, che crescono insieme."
        )
    else:
        summary.append(
            f"La relazione più forte è tra {max_corr[0]} e {max_corr[1]}, ma sono inversamente correlate."
        )

    # 3. VARIABILITÀ (solo se significativa)
    for stat in stats_ml:

        std = df_filtered[stat].std()
        mean = df_filtered[stat].mean()

        if std > mean * 0.6:
            insights.append(f"{stat} varia molto tra i Pokémon.")

    # 4. STAT PIÙ PREVEDIBILE
    predictability = {}

    for stat in stats_ml:
        correlations = corr[stat].drop(stat)
        predictability[stat] = correlations.abs().max()

    most_predictable = max(predictability, key=predictability.get)

    summary.append(
        f"La statistica più prevedibile è {most_predictable}, grazie alle forti correlazioni con le altre."
    )

    # OUTPUT
    for line in summary:
        st.write(f"- {line}")