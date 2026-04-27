# Pokémon Data Analysis & Machine Learning Dashboard

Progetto di data analysis e machine learning sviluppato in Python per esplorare, analizzare e interpretare le statistiche dei Pokémon.

L'applicazione permette di:
- analizzare dati statistici
- filtrare Pokémon per categorie e generazioni
- visualizzare correlazioni
- generare insight automatici
- predire statistiche tramite modelli di Machine Learning

---

## Demo

https://pokemon-dashboard-btkaws2e3aippkszerfdq8.streamlit.app/

---

## Funzionalità principali

### Overview
- KPI principali (media, massimo, distribuzione)
- istogrammi interattivi

### Top / Flop
- migliori e peggiori Pokémon per statistica
- informazioni dettagliate:
  - generazione
  - tipo
  - categoria (ordinario, leggendario, mitico, ultracreatura)

### Search
- ricerca Pokémon per nome
- visualizzazione pulita delle statistiche

### Type Analysis
- confronto tra tipi Pokémon
- medie e distribuzioni
- analisi avanzata: confronto tra type1, type2 e combinazione di entrambi

### Machine Learning
- predizione di statistiche (Attack, Defense, ecc.)
- modello: Random Forest Regressor
- metriche:
  - MAE (errore medio)
  - R² Score
- feature importance

### Correlation & Insights
- matrice di correlazione tra statistiche
- insight automatici:
  - relazioni tra variabili
  - migliori predittori
  - pattern nei dati

### AI Summary
- sintesi automatica del dataset:
  - relazioni principali
  - statistiche dominanti
  - predicibilità delle variabili

---

## Tecnologie utilizzate

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- Scikit-learn

---

## Installazione

1. Clona il repository:
git clone https://github.com/RockyT98/pokemon-dashboard.git
cd pokemon-dashboard

2. Crea ambiente virtuale:
python -m venv .venv

3. Attiva ambiente:
Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate

4. Installa dipendenze:
pip install -r requirements.txt

5. Avvia l’app:
streamlit run app.py

---

## Obiettivi del progetto

Questo progetto è stato sviluppato per:
- migliorare competenze in Data Analysis
- applicare Machine Learning su dataset reali
- costruire una dashboard interattiva
- trasformare dati in insight leggibili

---

## Autore

Progetto sviluppato da Rocco Tarantino