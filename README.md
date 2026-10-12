# Quantum Dynamics – Deep Learning for Forecasting Quantum Dynamics

Progetto per il corso di **Deep Learning** (Laurea Magistrale in Computer Science / Artificial Intelligence, Università della Calabria).

---

## 🎯 Obiettivo del Progetto

La simulazione classica di sistemi quantistici a molti corpi scala esponenzialmente con il numero di qubit a causa dell'entanglement, limitando l'osservazione a sistemi piccoli o a intervalli temporali brevi. 

L'obiettivo di questo progetto è impiegare modelli di **Deep Learning** per apprendere e prevedere l'evoluzione temporale (magnetizzazioni locali e correlazioni a due corpi) di una catena quantistica a **10 qubit** sotto il modello **PXP**, superando i limiti delle simulazioni tradizionali.

### 🔬 Le 3 Sfide Cardine
1. **Minimizzazione della Time Window**: Identificare la finestra temporale di osservazione minima in input in grado di garantire previsioni affidabili a lungo termine (rolling forecast).
2. **Super-Resolution Temporale**: Ricostruire la dinamica a risoluzioni temporali più elevate rispetto al campionamento originale.
3. **Explainability & Attention Maps**: Estrarre e interpretare le mappe di attenzione (modelli Transformer) per comprendere le correlazioni tra qubit e le dipendenze temporali apprese dal modello.

---

## 📊 Dataset

- **File**: `data/trajectories.csv` *(non versionato su Git per dimensioni, ~463 MB)*.
- **Fisica**: Dinamica del modello PXP su $N = 10$ qubit a partire da $400$ stati iniziali casuali.
- **Parametri temporali**: $dt = 0.02$, $T_{fin} = 20.0$, per un totale di $N_{points} = 1001$ passi temporali per traiettoria.
- **Formato (56 colonne)**:
  - `col 0`: tempo $t \in [0, 20]$
  - `col 1-10`: magnetizzazioni locali $(m_1, \dots, m_{10})$
  - `col 11-55`: correlazioni a due corpi $(c_{12}, c_{13}, \dots, c_{9,10})$ ($45$ correlazioni)
- **Dimensione complessiva**: $400 \times 1001 = 400.400$ righe.

---

## 🏗️ Architettura del Repository

```
Quantum Dynamics/
├── data/
│   ├── trajectories.csv        # Dataset grezzo (in .gitignore)
│   ├── trajectories.npy        # Tensore binario per caricamento istantaneo (in .gitignore)
│   └── Copia di README.txt     # Specifiche fisiche dei dati
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # Caricamento, verifiche fisiche, train/val/test split
│   ├── 02_windowing.ipynb             # Creazione finestre scorrevoli e griglia di valutazione
│   ├── 03_model_m1_lstm_direct.ipynb  # M1: LSTM multi-step diretto + tuning
│   ├── 04_model_m2_lstm_seq2seq.ipynb # M2: LSTM Encoder-Decoder + free-running eval
│   ├── 05_model_m3_transformer.ipynb  # M3: Transformer baseline
│   ├── 06_model_m4_transformer.ipynb  # M4: Transformer avanzato + Attention Maps (Explainability)
│   └── 07_results_comparison.ipynb    # Confronto metriche e grafici finali
├── src/
│   └── utils/
│       ├── __init__.py
│       ├── seeding.py          # Riproducibilità (seed fissati)
│       ├── data_loading.py     # Loader del dataset e partizionamento
│       ├── windowing.py        # Generazione dataset a finestre scorrevoli
│       ├── results_logger.py   # Logging standardizzato dei risultati
│       ├── metrics.py          # Metriche di valutazione (MSE, MAE)
│       └── viz.py              # Visualizzazioni e grafici scientifici (>=300 DPI)
├── results/
│   ├── checkpoints/            # Pesi dei modelli addestrati (in .gitignore)
│   └── figures/                # Grafici esportati per il report
├── report/                     # Documentazione e report finale
├── slides/                     # Presentazione finale
├── requirements.txt            # Dipendenze Python
└── .gitignore                  # Esclusioni file pesanti e temporanei
```

---

## 🚀 Setup e Installazione

### 1. Prerequisiti
- Python 3.12 (raccomandato per piena compatibilità con PyTorch e CUDA)
- Git

### 2. Creazione dell'Ambiente Virtuale
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📈 Progressioni dei Modelli

I modelli vengono sviluppati in progressione incrementale per risolvere gradualmente le criticità della dinamica quantistica:

| Modello | Architettura | Obiettivo | Motivazione / Risoluzione del limite precedente |
| :--- | :--- | :--- | :--- |
| **M1** | LSTM Diretto | Predizione single-shot su orizzonte multi-step | Baseline di riferimento, semplice da addestrare. Limite: non modella le correlazioni reciproche tra timestep futuri. |
| **M2** | LSTM Seq2Seq | Encoder-Decoder autoregressivo con valutazione free-running | Generazione sequenziale dei timestep futuri. Limite: bottleneck di contesto a vettore fisso. |
| **M3** | Transformer Baseline | Self-Attention su tutta la finestra temporale | Elimina il bottleneck di contesto e cattura dipendenze a lungo raggio. |
| **M4** | Transformer Spiegabile | Self-Attention + Estrazione Attention Maps | Risolve la sfida di **Explainability**: visualizza quali qubit e momenti passati governano l'evoluzione quantistica. |

---

## 📌 Regole e Best Practices del Progetto
- **Riproducibilità**: Seed fissati ovunque (`src/utils/seeding.py`).
- **Data Splitting per Traiettoria**: Split Train / Val / Test (es. 80% / 10% / 10%) effettuato a livello di intere traiettorie per scongiurare data leakage temporale.
- **Grafici ad Alta Risoluzione**: Tutte le figure esportate a $\ge 300\text{ DPI}$.
