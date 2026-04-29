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
    ["Overview", "Top/Flop", "Search", "Type Analysis", "Generational Analysis", "ML",
     "AI Insights", "Correlation & Insights"]
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
                "Total", "height", "weight"
            ]

            st.dataframe(result[cols], use_container_width=True)

#______________________________________________________________________________________
# TYPE ANALYSIS
#______________________________________________________________________________________
elif menu == "Type Analysis":

    st.title("Analisi Statistiche per Tipo Pokémon")

    stat = st.selectbox(
        "Statistica",
        ["attack", "defense", "speed", "hp", "sp_attack", "sp_defense", "Total"]
    )

    tipo_col = st.radio(
        "Tipo di analisi",
        ["type1", "type2", "Entrambi"]
    )

    # ------------------------------------------------------------------
    # PREPARAZIONE DATI
    # ------------------------------------------------------------------

    if tipo_col == "Entrambi":

        df_types = pd.concat([
            df_filtered[['name', 'type1', 'attack', 'defense', 'speed', 'hp',
                         'sp_attack', 'sp_defense', 'Total']].rename(columns={'type1': 'type'}),
            df_filtered[['name', 'type2', 'attack', 'defense', 'speed', 'hp',
                         'sp_attack', 'sp_defense', 'Total']].rename(columns={'type2': 'type'})
        ])

        df_types = df_types.dropna(subset=["type"])
        df_types = df_types[df_types["type"] != "None"]

    else:

        df_types = df_filtered[
            ['name', tipo_col, 'attack', 'defense', 'speed', 'hp',
             'sp_attack', 'sp_defense', 'Total']
        ].rename(columns={tipo_col: 'type'})

        df_types = df_types.dropna(subset=["type"])
        df_types = df_types[df_types["type"] != "None"]

    # ------------------------------------------------------------------
    # AGGREGAZIONE
    # ------------------------------------------------------------------

    tipo_stats = df_types.groupby("type")[stat].agg(["mean", "count"]).reset_index()
    tipo_stats["mean"] = tipo_stats["mean"].round(2)
    tipo_stats.sort_values(by="mean", ascending=False, inplace=True)

    # ------------------------------------------------------------------
    # GRAFICO MEDIA
    # ------------------------------------------------------------------

    st.subheader(f"Media {stat} per tipo")

    fig_mean = px.bar(
        tipo_stats,
        x="type",
        y="mean",
        text="mean",
        color="mean",
        color_continuous_scale=px.colors.sequential.Viridis
    )

    st.plotly_chart(fig_mean, use_container_width=True)

    # ------------------------------------------------------------------
    # GRAFICO COUNT
    # ------------------------------------------------------------------

    st.subheader("Numero Pokémon per tipo")

    fig_count = px.bar(
        tipo_stats,
        x="type",
        y="count",
        text="count",
        color="count",
        color_continuous_scale=px.colors.sequential.Plasma
    )

    st.plotly_chart(fig_count, use_container_width=True)

    # ------------------------------------------------------------------
    # INSIGHT GENERALI
    # ------------------------------------------------------------------

    max_type = tipo_stats.loc[tipo_stats["mean"].idxmax()]
    min_type = tipo_stats.loc[tipo_stats["mean"].idxmin()]

    st.markdown("---")

    col1, col2 = st.columns(2)

    col1.metric(
        "Miglior tipo",
        max_type["type"],
        str(max_type["mean"])
    )

    col2.metric(
        "Peggior tipo",
        min_type["type"],
        str(min_type["mean"])
    )

    # ------------------------------------------------------------------
    # ANALISI DETTAGLIATA PER TIPO
    # ------------------------------------------------------------------

    st.subheader("Analisi dettagliata per tipo")

    for t in sorted(df_types["type"].unique()):

        st.markdown("---")
        st.markdown(f"## Tipo: {t}")

        type_df = df_types[df_types["type"] == t]

        col1, col2 = st.columns(2)

        # =========================
        # TOTAL
        # =========================
        best_total = type_df.loc[type_df["Total"].idxmax()]
        worst_total = type_df.loc[type_df["Total"].idxmin()]

        st.markdown("### Total")

        st.write(f"Migliore: {best_total['name']} ({best_total['Total']})")
        st.write(f"Peggiore: {worst_total['name']} ({worst_total['Total']})")


        with col1:

            for stat_name in ["attack", "defense", "speed"]:
                best = type_df.loc[type_df[stat_name].idxmax()]
                worst = type_df.loc[type_df[stat_name].idxmin()]

                st.markdown(f"**{stat_name.upper()}**")

                st.write(f"Massimo: {best['name']} ({best[stat_name]})")
                st.write(f"Minimo: {worst['name']} ({worst[stat_name]})")

                st.write("---")


        with col2:

            for stat_name in ["sp_attack", "sp_defense", "hp"]:
                best = type_df.loc[type_df[stat_name].idxmax()]
                worst = type_df.loc[type_df[stat_name].idxmin()]

                st.markdown(f"**{stat_name.upper()}**")

                st.write(f"Massimo: {best['name']} ({best[stat_name]})")
                st.write(f"Minimo: {worst['name']} ({worst[stat_name]})")

                st.write("---")
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
# GENERATIONAL ANALYSIS
#______________________________________________________________________________________
elif menu == "Generational Analysis":
    st.header("Generational Analysis")

    # __________________________
    # GRAFICO MEDIA TOTAL PER GEN
    # __________________________

    gen_stats = df_filtered.groupby("generation")["Total"].mean().reset_index()

    fig1 = px.bar(
        gen_stats,
        x="generation",
        y="Total",
        title="Media Total per Generazione",
        color="generation",
        color_continuous_scale=px.colors.sequential.Plasma
    )

    fig1.update_layout(title_x=0.5, template="plotly_white", showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    # __________________________
    # NUMERO POKÉMON PER GEN
    # __________________________
    gen_count = df_filtered["generation"].value_counts().sort_index().reset_index()
    gen_count.columns = ["generation", "count"]

    fig2 = px.bar(
        gen_count,
        x="generation",
        y="count",
        title="Numero Pokémon per Generazione",
        color="count",
        color_continuous_scale=px.colors.sequential.Plasma
    )

    fig2.update_layout(title_x=0.5, template="plotly_white", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    #__________________________
    # INSIGHT PER GENERAZIONE
    #__________________________
    st.subheader("Analisi Dettagliata per Generazione")

    for gen in sorted(df_filtered["generation"].unique()):

        st.markdown("---")

        gen_df = df_filtered[df_filtered["generation"] == gen]

        st.markdown(f"### Generazione {gen}")

        col1, col2 = st.columns(2)
        # __________________________
        # TOTAL
        # __________________________
        best_total_gen = gen_df.loc[gen_df["Total"].idxmax()]
        worst_total_gen = gen_df.loc[gen_df["Total"].idxmin()]

        st.markdown("**Total**")
        st.write(f"Migliore: {best_total_gen['name']} ({best_total_gen['Total']})")
        st.write(f"Peggiore: {worst_total_gen['name']} ({worst_total_gen['Total']})")

        with col1:
            for stat_name in ["attack", "defense", "speed"]:

                best = gen_df.loc[gen_df[stat_name].idxmax()]
                worst = gen_df.loc[gen_df[stat_name].idxmin()]
                st.markdown(f"**{stat_name.upper()}**")

                st.write(f"Massimo: {best['name']} ({best[stat_name]})")
                st.write(f"Minimo: {worst['name']} ({worst[stat_name]})")

                st.write("---")

        with col2:
            for stat_name in ["sp_attack", "sp_defense","hp"]:

                best = gen_df.loc[gen_df[stat_name].idxmax()]
                worst = gen_df.loc[gen_df[stat_name].idxmin()]
                st.markdown(f"**{stat_name.upper()}**")

                st.write(f"Massimo: {best['name']} ({best[stat_name]})")
                st.write(f"Minimo: {worst['name']} ({worst[stat_name]})")

                st.write("---")


#______________________________________________________________________________________
# CORRELATION & INSIGHTS
#______________________________________________________________________________________
#______________________________________________________________________________________
# CORRELATION & INSIGHTS
#______________________________________________________________________________________

elif menu == "Correlation & Insights":

    st.header("Correlation Analysis & AI Insights")

    #____________________________________________________
    # CORRELATION MATRIX
    #____________________________________________________

    st.subheader("Correlation Matrix")

    corr = df_filtered[stats_ml].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Correlation between stats"
    )

    st.plotly_chart(fig, use_container_width=True)

    #____________________________________________________
    # CORRELATION INSIGHTS (CLASSICI)
    #____________________________________________________

    st.subheader("Correlation Insights")

    insights = []
    threshold = 0.6

    for col in corr.columns:
        for idx in corr.index:

            if col != idx:
                value = corr.loc[idx, col]

                if abs(value) >= threshold:

                    if value > 0:
                        insights.append(f"{col} e {idx} crescono insieme ({round(value,2)}).")
                    else:
                        insights.append(f"{col} cresce mentre {idx} diminuisce ({round(value,2)}).")

    #____________________________________________________
    # MIGLIORI PREDITTORI
    #____________________________________________________

    for target in stats_ml:

        correlations = corr[target].drop(target)

        best_feature = correlations.abs().idxmax()
        best_value = correlations[best_feature]

        if best_value > 0:
            insights.append(f"{best_feature} è il miglior indicatore per {target}.")
        else:
            insights.append(f"{best_feature} è inversamente legata a {target}.")

    #____________________________________________________
    # OUTPUT CORRELATIONS
    #____________________________________________________

    insights = list(set(insights))

    for text in insights[:12]:
        st.write(f"- {text}")

    #____________________________________________________
    # PATTERN PER GENERAZIONE
    #____________________________________________________

    st.subheader("Pattern per generazione")

    gen_stats = df_filtered.groupby("generation")[stats_ml].mean()

    for stat in stats_ml:

        best_gen = gen_stats[stat].idxmax()
        worst_gen = gen_stats[stat].idxmin()

        st.write(f"{stat.upper()}")
        st.write(f"- Migliore generazione: Gen {best_gen} ({round(gen_stats.loc[best_gen, stat], 2)})")
        st.write(f"- Peggiore generazione: Gen {worst_gen} ({round(gen_stats.loc[worst_gen, stat], 2)})")

    # ____________________________________________________
    # DISTRIBUZIONE TIPI PER GENERAZIONE
    # ____________________________________________________

    st.subheader("Distribuzione tipi per generazione")

    # prendiamo type1 e type2 insieme (più corretto)
    df_types = pd.concat([
        df_filtered[["generation", "type1"]].rename(columns={"type1": "type"}),
        df_filtered[["generation", "type2"]].rename(columns={"type2": "type"})
    ])

    df_types = df_types.dropna()
    df_types = df_types[df_types["type"] != "None"]

    # lista tipi unici
    all_types = sorted(df_types["type"].unique())

    for t in all_types:
        st.markdown("---")
        st.markdown(f"### Tipo: {t}")

        type_df = df_types[df_types["type"] == t]

        # conteggio per generazione
        gen_counts = type_df["generation"].value_counts().sort_index().reset_index()
        gen_counts.columns = ["generation", "count"]

        # generazioni con max e min presenza
        max_gen = gen_counts.loc[gen_counts["count"].idxmax()]
        min_gen = gen_counts.loc[gen_counts["count"].idxmin()]

        # GRAFICO
        fig = px.bar(
            gen_counts,
            x="generation",
            y="count",
            title=f"Distribuzione del tipo {t} per generazione",
            color="count",
            color_continuous_scale=px.colors.sequential.Viridis
        )

        fig.update_layout(
            title_x=0.5,
            template="plotly_white",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # INSIGHT TESTUALE
        col1, col2 = st.columns(2)

        col1.metric(
            "Generazione più presente",
            f"Gen {max_gen['generation']}",
            f"{max_gen['count']} Pokémon"
        )

        col2.metric(
            "Generazione meno presente",
            f"Gen {min_gen['generation']}",
            f"{min_gen['count']} Pokémon"
        )

    #____________________________________________________
    # INSIGHT GLOBALI
    #____________________________________________________

    st.subheader("Insight globali")

    # Pokémon più forti per generazione
    for gen in sorted(df_filtered["generation"].unique()):

        gen_df = df_filtered[df_filtered["generation"] == gen]

        strongest = gen_df.loc[gen_df["Total"].idxmax()]

        st.write(
            f"Gen {gen}: Pokémon più forte = {strongest['name']} ({strongest['Total']})"
        )

    # Tipo più comune globale
    type_counts = df_filtered["type1"].value_counts()

    st.write(
        f"Tipo più comune nel dataset: {type_counts.idxmax()} ({type_counts.max()} Pokémon)"
    )