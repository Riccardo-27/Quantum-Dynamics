<div align="center">

# ⚛️ Quantum Dynamics Forecasting
### Deep Learning Surrogate Models for Many-Body Quantum Systems Beyond the Classical Simulation Barrier

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12/13](https://img.shields.io/badge/CUDA-Enabled-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Status](https://img.shields.io/badge/Project-Active%20Research-blueviolet?style=flat)](#)
[![University of Calabria](https://img.shields.io/badge/UniCal-Deep%20Learning%202025%2F2026-B81E24?style=flat)](https://www.unical.it/)

<p align="center">
  <b>Unsupervised & Sequence-to-Sequence Modeling of PXP Quantum Dynamics</b><br>
  <i>Previsione della dinamica quantistica di magnetizzazioni e correlazioni a molti corpi mediante architetture ricorrenti e Transformer spiegabili.</i>
</p>

</div>

---

## 📑 Indice

1. [Visione Scientifica e Motivazione](#-visione-scientifica-e-motivazione)
2. [Fondamenti Fisici e Formulazione Matematica](#-fondamenti-fisici-e-formulazione-matematica)
3. [Le 3 Sfide Cardine della Ricerca](#-le-3-sfide-cardine-della-ricerca)
4. [Dataset e Struttura dei Dati](#-dataset-e-struttura-dei-dati)
5. [Architettura Incrementale dei Modelli](#-architettura-incrementale-dei-modelli)
6. [Protocollo Sperimentale e Valutazione Free-Running](#-protocollo-sperimentale-e-valutazione-free-running)
7. [Struttura del Repository e Moduli](#-struttura-del-repository-e-moduli)
8. [Guida Rapida all'Installazione e Uso](#-guida-rapida-allinstallazione-e-uso)
9. [Riproducibilità e Best Practices](#-riproducibilit%C3%A0-e-best-practices)
10. [Riferimenti e Autori](#-riferimenti-e-autori)

---

## 🌌 Visione Scientifica e Motivazione

Nei dispositivi quantistici a tecnologia atomica neutra (array di atomi di Rydberg) e nei processori quantistici ad alto numero di qubit ($N \sim 100$), i singoli qubit interagiscono secondo leggi non-classiche generando **entanglement a molti corpi**.

```
  +--------------------------------------------------------------------------------+
  |                        IL MURO DELLA SIMULAZIONE CLASSICA                      |
  |                                                                                |
  |   Simulazione Esatta: Dimensione dello Spazio di Hilbert: dim(H) = 2^N        |
  |   N = 10  -> 1.024 stati        (Trattabile con calcolo esatto classico)       |
  |   N = 30  -> ~10^9 stati        (Limite della RAM dei cluster HPC)             |
  |   N = 100 -> ~10^30 stati       (IMPOSSIBILE per qualsiasi supercomputer)      |
  |                                                                                |
  |   Tensor Networks (DMRG/TEBD): Limitati dalla crescita dell'entropia di       |
  |   entanglement S(t) ~ v_E * t  --> Il bond dimension chi esplode nel tempo.   |
  +--------------------------------------------------------------------------------+
```

### La Soluzione del Deep Learning
Invece di calcolare l'intero vettore di stato $|\psi(t)\rangle$ nello spazio di Hilbert esponenziale, addestriamo reti neurali ad **apprendere la dinamica effettiva delle osservabili fisiche rilevanti a bassa dimensione** (magnetizzazioni e correlazioni a due corpi).

In questo modo il modello funge da **Quantum Surrogate Model**: osserva una sequenza iniziale breve ottenuta con simulazioni tradizionali e **prevede l'evoluzione futura ad orizzonti temporali molto più lunghi**, aprendo la strada all'interpretazione del comportamento di chip quantistici altrimenti inaccessibili (la cosiddetta *Quantum Black Box*).

---

## 🔬 Fondamenti Fisici e Formulazione Matematica

### 1. Il Modello PXP (Rydberg Blockade)
Il sistema simulato nel dataset è modellato dall'Hamiltoniana **PXP** per una catena monodimensionale di $N = 10$ qubit con condizioni al contorno periodiche o aperte:

$$H = \sum_{j=1}^{N} P_{j-1} X_j P_{j+1}$$

dove:
- $X_j = \sigma_j^x$ è l'operatore di Pauli di spin-flip che tenta di eccitare il qubit $j$.
- $P_j = |0\rangle\langle 0|_j = \frac{I + \sigma_j^z}{2}$ è il proiettore sullo stato fondamentale (ground state).

> **Vincolo Fisico (Rydberg Blockade)**: Un qubit può cambiare stato solo se entrambi i suoi primi vicini si trovano nello stato fondamentale $|0\rangle$. Questo vincolo dinamico genera fenomeni quantistici complessi, inclusi i *Quantum Many-Body Scars* e l'assenza di termalizzazione rapida.

### 2. Le 55 Osservabili Tracciate
A ogni istante di tempo $t$, il sistema misura:
1. **10 Magnetizzazioni Locali ($m_j$)**:
   $$m_j(t) = \langle \psi(t) | \sigma_j^z | \psi(t) \rangle \in [-1, 1], \quad j \in \{1, \dots, 10\}$$
2. **45 Correlazioni Connesse a Due Corpi ($c_{jk}$)**:
   $$c_{jk}(t) = \langle \psi(t) | \sigma_j^z \sigma_k^z | \psi(t) \rangle - m_j(t) m_k(t), \quad 1 \le j < k \le 10$$
   Il numero di coppie uniche è $\binom{10}{2} = \frac{10 \times 9}{2} = 45$.

Ogni stato quantistico a tempo $t$ è dunque un vettore:
$$\mathbf{s}(t) = \big[ m_1(t), \dots, m_{10}(t), c_{1,2}(t), \dots, c_{9,10}(t) \big]^T \in \mathbb{R}^{55}$$

---

## 🎯 Le 3 Sfide Cardine della Ricerca

Il progetto affronta formalmente le tre sfide introdotte nei requisiti ufficiali:

```
+--------------------------------------------------------------------------------------------------+
| SFIDA 1: MINIMA TIME WINDOW                                                                     |
| Identificare la finestra temporale minima di osservazione T_in che massimizza la precisione      |
| a lungo termine T_out. Analisi di sensibilità su griglia: T_in in {50, 100, 200} dt.              |
+--------------------------------------------------------------------------------------------------+
                                                |
                                                v
+--------------------------------------------------------------------------------------------------+
| SFIDA 2: SUPER-RESOLUTION TEMPORALE                                                             |
| Ricostruire traiettorie continue o a campionamento fine (dt_fine = dt / k) a partire da          |
| traiettorie campionate a bassa frequenza (dt_coarse), modellando la fisica tra i timestep.      |
+--------------------------------------------------------------------------------------------------+
                                                |
                                                v
+--------------------------------------------------------------------------------------------------+
| SFIDA 3: EXPLAINABILITY & ATTENTION MAPS                                                         |
| Decodificare la dinamica quantistica attraverso le mappe di auto-attenzione del Transformer:    |
| - Relazione tra peso di attenzione e distanza spaziale tra qubit |j - k|.                        |
| - Propagazione dell'informazione e limiti conici di Lieb-Robinson nel tempo.                    |
+--------------------------------------------------------------------------------------------------+
```

---

## 📊 Dataset e Struttura dei Dati

- **File Sorgente**: `data/trajectories.csv` (~463.8 MB, 400.400 righe totali).
- **Rappresentazione Binaria Ottimizzata**: `data/trajectories.npy` (**84 MB**, float32) caricabile in **meno di 50 ms**.
- **Parametri di Simulazione**:
  - Numero di traiettorie ($N_{traj}$): **400** (generate da stati quantistici casuali indipendenti).
  - Punti temporali per traiettoria ($N_{points}$): **1001** passi.
  - Orizzonte temporale: $t \in [0.0, 20.0]$ con campionamento costante $dt = 0.02$.
  - Feature per timestep: **55** variabili fisiche continue (tempo escluso dalle feature).

### Partizionamento a Traiettoria Isolata (Anti-Data Leakage)
A differenza delle serie temporali convenzionali, **lo split non viene mai effettuato lungo l'asse temporale** della stessa simulazione. Si applica un partizionamento rigoroso sull'indice della traiettoria:

$$\text{Train (80\%)}: 320 \text{ Traiettorie} \quad\Big|\quad \text{Validation (10\%)}: 40 \text{ Traiettorie} \quad\Big|\quad \text{Test (10\%)}: 40 \text{ Traiettorie}$$

---

## 📈 Architettura Incrementale dei Modelli

La ricerca adotta una metodologia rigorosa: **nessun modello è fine a se stesso**. Ciascuna architettura risolve un limite teorico e pratico del modello precedente:

```
[M1: LSTM Diretto Multi-Step]
   |
   +--> Limite: Le predizioni future sono indipendenti; nessuna correlazione tra t+1 e t+H.
   v
[M2: LSTM Encoder-Decoder Seq2Seq]
   |    - Autoregressive Decoding con Scheduled Sampling
   |    - Valutazione in modalità Free-Running pura
   |
   +--> Limite: Bottleneck del vettore di contesto nascosto fisso (h_T, c_T).
   v
[M3: Transformer Baseline]
   |    - Self-Attention globale su tutta la finestra T_in
   |    - Nessun collo di bottiglia e perfetta cattura di dipendenze a lungo raggio
   |
   +--> Limite: Scatola nera; nessuna analisi delle matrici di attenzione o super-resolution.
   v
[M4: Explainable Quantum Transformer]
        - Estrazione e visualizzazione delle Attention Maps (Sfida 3)
        - Head per Super-Resolution temporale continua (Sfida 2)
        - Valutazione comparativa finale con metriche scientifiche
```

### Tabella Comparativa delle Architetture

| Modello | Famiglia | Paradigma di Predizione | Vantaggi Chiave | Limite Principale che Motiva lo Step Successivo |
| :--- | :--- | :--- | :--- | :--- |
| **M1** | RNN | Single-Shot Projection $\mathbb{R}^{T_{in} \times 55} \to \mathbb{R}^{T_{out} \times 55}$ | Addestramento rapido, nessun accumulo autoregressivo d'errore | Non modella le interdipendenze fisiche tra i singoli passi futuri |
| **M2** | RNN | Autoregressive Decoder con Scheduled Sampling | Modella la causalità temporale passo-passo; consente rollout indefiniti | Collo di bottiglia dello stato nascosto fisso; decadimento su lunghi orizzonti |
| **M3** | Attention | Self-Attention causale con Positional Encoding | Elimina il bottleneck di contesto; accesso diretto a qualunque timestep passato | Non sfrutta le mappe di attenzione per spiegare la connettività spaziale tra qubit |
| **M4** | Hybrid / Explainable | Transformer + Attention Extraction + Super-Resolution | **Spiega la propagazione dell'entanglement** e riscostruisce punti a risoluzione fine | Maggiore complessità parametrica e costo di tuning |

---

## 🧪 Protocollo Sperimentale e Valutazione Free-Running

### Il Protocollo a 3 Fasi per Modelli Ricorrenti
1. **Teacher Forcing (TF)**: Durante le prime epoche, il decoder riceve in input il ground truth $y_{t-1}^*$ per stabilizzare i gradienti.
2. **Scheduled Sampling (SS)**: La probabilità di fornire il valore reale o il valore predetto decade gradualmente: $p_{TF}(\text{epoch}) \to 0$.
3. **Free-Running Rollout (Valutazione Pura)**: In fase di validazione e test, il modello **non riceve alcun aiuto esterno**: ogni predizione alimenta ricorsivamente il passo successivo:

$$\hat{\mathbf{s}}_{t+1} = f_\theta(\hat{\mathbf{s}}_t, \hat{\mathbf{s}}_{t-1}, \dots)$$

### Metriche di Valutazione Standardizzate
Tutti gli esperimenti calcolano:
- **Mean Squared Error (MSE)** complessivo e per orizzonte:
  $$\text{MSE}(h) = \frac{1}{N_{test} \cdot 55} \sum_{i=1}^{N_{test}} \sum_{f=1}^{55} \big( s_{i, T_{in} + h, f} - \hat{s}_{i, T_{in} + h, f} \big)^2$$
- **Mean Absolute Error (MAE)**.
- **Root Mean Squared Error (RMSE)**.
- **Physical Boundary Check**: Verifica che le predizioni rimangano nei limiti fisici $m_j \in [-1, 1]$.

---

## 📂 Struttura del Repository e Moduli

Il codice è organizzato secondo standard accademici per separare logica riusabile, pipeline sperimentali e reportistica:

```
Quantum Dynamics/
├── data/
│   ├── trajectories.csv               # Dataset grezzo simulato (463.8 MB, in .gitignore)
│   ├── trajectories.npy               # Tensore binario ottimizzato (84 MB, in .gitignore)
│   ├── time_grid.npy                  # Vettore dei 1001 istanti temporali
│   └── Copia di README.txt            # Documentazione fisica originaria
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # Analisi esplorativa, fisica delle feature, split
│   ├── 02_windowing.ipynb             # Configurazione finestre scorrevoli (Sfida 1)
│   ├── 03_model_m1_lstm_direct.ipynb  # M1: LSTM Multi-Step Diretto
│   ├── 04_model_m2_lstm_seq2seq.ipynb # M2: LSTM Seq2Seq + Scheduled Sampling + Free-Running
│   ├── 05_model_m3_transformer.ipynb  # M3: Transformer Baseline
│   ├── 06_model_m4_transformer.ipynb  # M4: Explainability (Attention Maps) + Super-Resolution
│   └── 07_results_comparison.ipynb    # Sintesi finale, tabelle prestazionali e report
│
├── src/
│   └── utils/
│       ├── seeding.py                 # Fissaggio semi per riproducibilità deterministica
│       ├── data_loading.py            # I/O rapido e partizionamento a traiettoria
│       ├── windowing.py               # Dataset sliding window e PyTorch DataLoader
│       ├── results_logger.py          # Logger atomico su results_log.csv
│       ├── metrics.py                 # Calcolo MSE, MAE, RMSE e decadimento per orizzonte
│       └── viz.py                     # Grafici scientifici ad alta risoluzione (>= 300 DPI)
│
├── results/
│   ├── checkpoints/                   # Pesi dei modelli addestrati (in .gitignore)
│   ├── figures/                       # Grafici esportati ad alta risoluzione per la tesi/report
│   └── results_log.csv                # Tabella condivisa con tutte le run e gli iperparametri
│
├── report/                            # Report scientifico finale (PDF / LaTeX)
├── slides/                            # Slide per la presentazione orale
├── requirements.txt                   # Dipendenze con versioni bloccate
├── .gitignore                         # Esclusione di dati pesanti, venv e file temporanei
└── README.md                          # Questo documento
```

---

## 🚀 Guida Rapida all'Installazione e Uso

### 1. Clonazione del Repository
```bash
git clone https://github.com/Riccardo-27/Quantum-Dynamics.git
cd Quantum-Dynamics
```

### 2. Creazione dell'Ambiente Virtuale
```powershell
# Su Windows con Python 3.12:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Preprocessing Immediato dei Dati
Il caricamento iniziale dal CSV viene convertito in modo trasparente dal modulo [`src/utils/data_loading.py`](file:///c:/Users/Riccardo/Desktop/Deep%20Learning/Quantum%20Dynamics/src/utils/data_loading.py):

```python
import sys
sys.path.append("src")
from utils.data_loading import load_dataset, split_trajectories

# Caricamento istantaneo (84 MB in RAM)
features, time_grid = load_dataset("data")
print("Dimensioni Dataset:", features.shape)  # (400, 1001, 55)

# Split privo di data leakage
train_idx, val_idx, test_idx = split_trajectories(num_trajectories=400, seed=42)
```

---

## 🛡️ Riproducibilità e Best Practices

1. **Semi Casuali Fissati**: Ogni esperimento richiama [`set_seeds(42)`](file:///c:/Users/Riccardo/Desktop/Deep%20Learning/Quantum%20Dynamics/src/utils/seeding.py) per garantire determinismo su CPU, GPU (cuDNN deterministico) e generatori pseudocasuali.
2. **Archiviazione Figure ad Alta Fedeltà**: Tutti i plot vengono salvati tramite [`src/utils/viz.py`](file:///c:/Users/Riccardo/Desktop/Deep%20Learning/Quantum%20Dynamics/src/utils/viz.py) a **$\ge 300\text{ DPI}$** in formato `.jpeg` / `.png`.
3. **Log Centralizzato e Trasparente**: Tutte le run di addestramento salvano le metriche su [`results/results_log.csv`](file:///c:/Users/Riccardo/Desktop/Deep%20Learning/Quantum%20Dynamics/results/results_log.csv), rendendo replicabile e verificabile ogni riga della tabella riassuntiva.

---

## 👥 Riferimenti e Contesto Accademico

- **Corso**: Deep Learning (9 CFU), A.Y. 2025/2026.
- **Laurea Magistrale**: Computer Science / Artificial Intelligence, **Università della Calabria (UniCal)**.
- **Docenti di Riferimento**: Prof. Gianluigi Greco, Dott. Carlo Adornetto.
- **Traccia**: *Quantum Dynamics – Forecasting PXP Model Dynamics*.
- **Autore**: Riccardo ([@Riccardo-27](https://github.com/Riccardo-27)).
