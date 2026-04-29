# Pokémon Data Analysis & Machine Learning Dashboard

Progetto di data analysis e machine learning sviluppato in Python per esplorare, analizzare e interpretare le statistiche dei Pokémon tramite una dashboard interattiva.

L’app permette di trasformare un dataset Pokémon in insight visivi, statistici e predittivi usando Streamlit, Plotly e Scikit-learn.

---

## Demo

https://pokemon-dashboard-btkaws2e3aippkszerfdq8.streamlit.app/

---

## Funzionalità principali

### Overview
- KPI principali (media, massimo, distribuzione)
- analisi rapide delle statistiche Pokémon
- istogrammi interattivi

---

### Top / Flop
- migliori e peggiori Pokémon per ogni statistica
- dettagli completi:
  - generazione
  - tipo
  - categoria (ordinario, leggendario, mitico, ultracreatura)

---

### Search
- ricerca Pokémon per nome
- visualizzazione completa delle statistiche

---

### Type Analysis
- confronto tra tipi Pokémon
- analisi avanzata:
  - type1
  - type2
  - combinazione dei due
- medie, distribuzioni e ranking per tipo

---

### Generational Analysis
- confronto tra generazioni Pokémon
- Pokémon più forti e più deboli per generazione
- distribuzione delle statistiche nel tempo

---

### Machine Learning
- predizione statistiche Pokémon (Attack, Defense, Speed, ecc.)
- modello: Random Forest Regressor
- metriche:
  - MAE (errore medio assoluto)
  - R² Score
- feature importance delle variabili

---

### Correlation & Insights
- matrice di correlazione tra statistiche
- insight automatici:
  - relazioni tra variabili
  - pattern nascosti nei dati
  - migliori predittori per ogni statistica

---

### AI Summary
- sintesi automatica del dataset:
  - statistiche più variabili
  - relazioni principali
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

### 1. Clona repository
```bash
git clone https://github.com/RockyT98/pokemon-dashboard.git
cd pokemon-dashboard
```

### 2. Crea ambiente virtuale
```bash
python -m venv .venv
```

### 3. Clona repository
#### Windows
```bash
.venv\Scripts\activate
```
#### Mac/Linux:
```bash
source .venv/bin/activate
```
### 4. Installa dipendenze
```bash
pip install -r requirements.txt
```
### 5. Avvia app
```bash
streamlit run app.py
```
---

## Obiettivo del Progetto
Questo progetto è stato sviluppato per:

- applicare Data Analysis su dataset reali
- costruire una dashboard interattiva completa
- integrare Machine Learning in un sistema reale
- trasformare dati complessi in insight leggibili

---
## Autore
Progetto sviluppato da Rocco Tarantino
```bash
GitHub: https://github.com/RockyT98
```




