# TS-GNN Project Status
*Last Updated: 2026-04-05*

> **Scopo di questo file:** Mantenere un registro dettagliato di cosa è stato fatto, le decisioni architetturali prese e i prossimi step. Usalo come punto di partenza per ogni nuova sessione per evitare di sprecare token nel recappaggio del contesto.

## 🎯 Obiettivo Corrente (Fase 1)
Migrare l'infrastruttura del progetto **TS-GNN (Temporal Sheaf Graph Neural Networks per l'evoluzione del tumore TP53-driven)** verso **Google Drive / Google Colab**.
*   **Focus Biologico Principale:** Cellule di Cancro al Seno (TNBC / Basal-like) per l'addestramento (garantisce misurazioni TP53 molto più pulite rispetto al CRC).
*   **Focus Zero-Shot:** CRC (Colorectal Cancer) verrà usato solo come validazione di generalizzazione (zero-shot transfer).

## ✅ Cosa è stato completato
1.  **Analisi del Proposal (`tp53_brca_proposal.docx`)**:
    *   Letto e recepito il cambio di focus dal CRC al Breast Cancer (Gambardella 2022, GSE176078, GSE158508).
2.  **Refactoring dei Path per Google Drive**:
    *   Tutti i path hardcoded (`D:/tsgnn_data`) nei file della pipeline (`src/tsgnn/data/download.py`, `allele_embeddings.py`, `debug_brca.py`) sono stati sostituiti con path dinamici relativi alla directory di lavoro (`Path(os.getcwd()) / "data"`). In questo modo, quando girano su Colab, puntano automaticamente al tuo Google Drive.
    *   La cache di PyTorch (`TORCH_HOME`) per i grandi modelli ESM è stata reindirizzata alla cartella di lavoro per non intasare lo spazio di root di Colab.
3.  **Creazione Script per Colab (`Colab_Data_Downloader.ipynb`)**:
    *   Ho creato un notebook Jupyter ottimizzato per Colab che esegue automaticamente il mounting di Google Drive, l'installazione delle dipendenze (`scanpy`, `fair-esm`, ecc.) e il download massivo di tutti i dataset (incluso Gambardella 2022 dall'API di Figshare) direttamente nella cartella `./data/` su Drive.

## 🚧 A che punto siamo ORA
Il codice per il data-download in cloud è pronto per essere eseguito da te. Tutta la struttura locale è stata convertita in una struttura "Cloud/Drive First".

## 🔜 Prossimi Step (What to do next)

### Azioni richieste all'Utente (SUBITO):
1.  **Carica** l'intera cartella `ts-gnn` sul tuo Google Drive (ad esempio dentro `I miei file/Sheaf Neural Networks/ts-gnn`).
2.  **Apri** il file `Colab_Data_Downloader.ipynb` con Google Colab.
3.  **Modifica** se necessario il path di root nella prima cella (es: `ROOT_DIR = '/content/drive/MyDrive/Sheaf Neural Networks/ts-gnn'`).
4.  **Esegui (Run All)**: Questo scaricherà i dataset da 10+ GB (Gambardella, Pelka, ecc.) direttamente e unicamente sul tuo storage Google Drive.

### Azioni che farà Gemini (DOPO il download):
Una volta che i dati saranno fisicamente sul tuo Drive, la vera pipe di Machine Learning inizierà:
1.  **Dataloader creation:** Scrivere la logica in `scanpy` per leggere i file `RAW.UMI.counts.BC.cell.lines.rds` (o mtx) di Gambardella in oggetti `AnnData`.
2.  **Preprocessing del Grafo:** Convertire l'espressione genica delle singole cellule in **Grafi** in base alle reti di co-espressione.
3.  **Integrazione TS-GNN:** Far "digerire" questi file al modello TS-GNN, applicando l'allele TP53 mutato come condizione (embedding ESM).
