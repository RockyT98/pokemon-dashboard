import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Pokemon Dashboard", layout="wide")

# Caricamento dati
@st.cache_data
def load_data():
    df = pd.read_csv("data/pokemon.csv")
    df["type2"] = df["type2"].fillna("None")
    df.rename(columns={"height (m)": "height", "weight (kg)": "weight"}, inplace=True)
    return df

df = load_data()

# -------------------------
# SIDEBAR MENU
# -------------------------
st.sidebar.title("Menu")

menu = st.sidebar.radio(
    "Navigazione",
    ["Overview", "Top / Flop", "Ricerca Pokémon", "Analisi per Tipo"]
)

# -------------------------
# FILTRI GLOBALI
# -------------------------
st.sidebar.header("Filtri")

include_legendary = st.sidebar.checkbox("Includi Leggendari", value=True)
include_mythical = st.sidebar.checkbox("Include Pokemon Misteriosi", value=True)
include_ultrabeast = st.sidebar.checkbox("Include Ultracreature", value=True)
include_ordinary = st.sidebar.checkbox("Include Pokemon Ordinari", value=True)

generations = sorted(df["generation"].unique())
selected_gen = st.sidebar.multiselect(
    "Generazione",
    options=generations,
    default=generations
)

df_filtered = df.copy()

if not include_legendary:
    df_filtered = df_filtered[df_filtered["is_legendary"] == 0]

if not include_mythical:
    df_filtered = df_filtered[df_filtered["is_mythical"] == 0]

if not include_ultrabeast:
    df_filtered = df_filtered[df_filtered["is_ultrabeast"] == 0]

if not include_ordinary:
    df_filtered = df_filtered[df_filtered["is_ordinary"] == 0]

if selected_gen:
    df_filtered = df_filtered[df_filtered["generation"].isin(selected_gen)]

if df_filtered.empty:
    st.warning("Hai filtrato tutto!")
    st.info("Suggerimento: attiva almeno una tra Ordinari, Leggendari, Misteriosi o Ultracreature.")
    st.stop()

# Statistiche disponibili
stats = ["attack", "defense", "speed", "sp_attack", "sp_defense", "hp", "Total", "height", "weight"]

# -------------------------
# OVERVIEW
# -------------------------
if menu == "Overview":
    st.title("Pokemon Dashboard - Overview")

    stat = st.selectbox("Seleziona statistica", stats)
    st.markdown("---")

    # Funzione helper per formattare i nomi dei Pokémon
    def format_pokemon(df, max_display=2):
        names = [f"{row['name'].title()} (Gen {row['generation']})" for _, row in df.iterrows()]
        if len(names) > max_display:
            return ", ".join(names[:max_display]) + f" +{len(names) - max_display} altri"
        return ", ".join(names)

    # Calcolo KPI
    total_pokemon = len(df_filtered)
    mean_value = round(df_filtered[stat].mean(), 2)

    max_value = df_filtered[stat].max()
    min_value = df_filtered[stat].min()

    max_pokemon = df_filtered[df_filtered[stat] == max_value][["name", "generation"]]
    min_pokemon = df_filtered[df_filtered[stat] == min_value][["name", "generation"]]

    max_names = format_pokemon(max_pokemon, max_display=2)
    min_names = format_pokemon(min_pokemon, max_display=2)

    # KPI in colonne
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Totale Pokémon", total_pokemon)
    col2.metric("Media", mean_value)
    col3.metric("Valore massimo", max_value, delta=max_names)
    col4.metric("Valore minimo", min_value, delta=min_names)

    st.markdown("---")

    # Istogramma
    st.subheader("Distribuzione statistica")
    fig_hist = px.histogram(
        df_filtered,
        x=stat,
        nbins=30,
        title=f"Distribuzione di {stat}",
        color_discrete_sequence=["#1f77b4"]
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Tabella dati filtrati
    with st.expander("Mostra dati filtrati"):
        st.dataframe(df_filtered.sort_values(by=stat, ascending=False), use_container_width=True)

# -------------------------
# TOP / FLOP
# -------------------------
elif menu == "Top / Flop":
    st.title("Top e Flop Pokémon")

    stat = st.selectbox("Seleziona statistica", stats)

    top = df_filtered.sort_values(by=stat, ascending=False).head(10)
    flop = df_filtered.sort_values(by=stat, ascending=True).head(10)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10")
        fig_top = px.bar(top, x="name", y=stat, color=stat)
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        st.subheader("Flop 10")
        fig_flop = px.bar(flop, x="name", y=stat, color=stat)
        st.plotly_chart(fig_flop, use_container_width=True)

# -------------------------
# RICERCA POKEMON
# -------------------------
elif menu == "Ricerca Pokémon":
    st.title("Ricerca Pokémon")

    nome = st.text_input("Inserisci nome")

    if nome:
        result = df_filtered[df_filtered["name"].str.contains(nome, case=False, na=False)]

        if result.empty:
            st.warning("Nessun Pokémon trovato")
        else:
            st.dataframe(result)

elif menu == "Analisi per Tipo":
    st.title("Analisi Statistiche per Tipo Pokémon")

    stat = st.selectbox("Seleziona statistica", stats)
    tipo_col = st.radio("Scegli tipo da analizzare", ["type1", "type2", "Entrambi"])

    # -------------------------
    # Prepara dataframe per tipo
    # -------------------------
    if tipo_col == "Entrambi":
        # Concateniamo type1 e type2
        df_types = pd.concat([
            df_filtered[['type1', stat]].rename(columns={'type1':'type'}),
            df_filtered[['type2', stat]].rename(columns={'type2':'type'})
        ])

        # Escludiamo valori None/NaN per la media
        df_types = df_types[df_types['type'].notna() & (df_types['type'] != "None")]

    else:
        df_types = df_filtered[[tipo_col, stat]].rename(columns={tipo_col:'type'})

        if tipo_col == "type1":
            # type1 non può essere None, ma nel caso filtriamo
            df_types = df_types[df_types['type'].notna() & (df_types['type'] != "None")]
        elif tipo_col == "type2":
            # Per type2, sostituiamo None con "Puro"
            df_types['type'] = df_types['type'].fillna("Puro")
            df_types.loc[df_types['type'] == "None", 'type'] = "Puro"

    # -------------------------
    # Raggruppamento
    # -------------------------
    tipo_stats = df_types.groupby('type')[stat].agg(['mean','count']).reset_index()
    tipo_stats['mean'] = tipo_stats['mean'].round(2)  # Arrotondiamo a 2 decimali
    tipo_stats.sort_values(by='mean', ascending=False, inplace=True)

    # -------------------------
    # Istogramma della media
    # -------------------------
    st.subheader(f"Media di {stat} per tipo")
    fig_bar = px.bar(
        tipo_stats[tipo_stats['type'] != "Puro"],  # Non consideriamo "Puro" nella media
        x='type',
        y='mean',
        text='mean',
        color='mean',
        color_continuous_scale=px.colors.sequential.Viridis,
        title=f"Media di {stat} per tipo"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # -------------------------
    # Istogramma del conteggio
    # -------------------------
    st.subheader(f"Quantità di Pokémon per tipo")
    fig_count = px.bar(
        tipo_stats,
        x='type',
        y='count',
        text='count',
        color='count',
        color_continuous_scale=px.colors.sequential.Plasma,
        title="Numero di Pokémon per tipo"
    )
    st.plotly_chart(fig_count, use_container_width=True)

    # -------------------------
    # Mostra tipo con max/min media
    # -------------------------
    # Ignoriamo "Puro" per media
    tipo_stats_media = tipo_stats[tipo_stats['type'] != "Puro"]
    max_tipo = tipo_stats_media.loc[tipo_stats_media['mean'].idxmax()]
    min_tipo = tipo_stats_media.loc[tipo_stats_media['mean'].idxmin()]

    st.markdown(f"**Tipo con {stat} medio più alto:** {max_tipo['type']} ({round(max_tipo['mean'],2)})")
    st.markdown(f"**Tipo con {stat} medio più basso:** {min_tipo['type']} ({round(min_tipo['mean'],2)})")