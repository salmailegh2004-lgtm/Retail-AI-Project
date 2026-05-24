# Retail AI Intelligence

Projet Python pour l'analyse retail avec un systeme multi-agents. Il permet de preparer des donnees CSV, lancer des modeles LSTM, detecter des pics de demande ou des anomalies, generer des recommandations business et exporter un rapport final.

## Fonctionnalites

- Pretraitement automatique des fichiers CSV retail.
- Forecasting de la demande et detection des periodes de pic.
- Detection d'anomalies avec un autoencoder LSTM.
- Generation de recommandations business.
- Validation humaine des recommandations.
- Export de rapports en TXT, JSON et PDF.
- Interface Streamlit pour executer et consulter les resultats.

## Structure

```text
projetDL/
├── app.py                  # Interface Streamlit
├── main.py                 # Execution console du pipeline
├── train_pipeline.py       # Entrainement des modeles
├── requirements.txt        # Dependances Python principales
├── agents/                 # Agents du pipeline
├── data/store_sales/       # Donnees retail d'exemple
├── outputs/                # Modeles, resultats et rapports generes
└── uploaded_data/          # Fichiers importes via l'interface
```

## Installation

Depuis le dossier `projetDL` :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install streamlit reportlab
```

`streamlit` et `reportlab` sont utilises par `app.py` pour l'interface et l'export PDF.

## Entrainement des modeles

Avant l'inference, les modeles doivent exister dans `outputs/`. Pour les generer :

```bash
python train_pipeline.py
```

Cette commande prepare les donnees, entraine le modele de forecasting et le modele de detection d'anomalies, puis sauvegarde les artefacts dans `outputs/`.

## Lancement en console

```bash
python main.py
```

Le script demande :

- le mode d'analyse : forecasting ou anomalie ;
- le chemin du dataset, par defaut `data/store_sales`.

## Lancement de l'interface

```bash
streamlit run app.py
```

L'interface permet de :

- choisir le mode d'analyse ;
- importer des fichiers CSV ;
- lancer le pipeline ;
- visualiser les metriques ;
- valider les recommandations ;
- generer et telecharger le rapport final.

## Fichiers de sortie importants

Les resultats sont sauvegardes dans `outputs/`, notamment :

- `pipeline_output.json`
- `orchestrator_logs.json`
- `forecasting_summary.json`
- `anomaly_summary.json`
- `recommendation_report.json`
- `human_validation_report.json`
- `final_business_report.txt`
- `final_business_report.json`
- `final_business_report.pdf`

## Notes

- Le pipeline utilise les modeles sauvegardes dans `outputs/` pour l'inference.
- Si les fichiers de modele sont manquants, lancez d'abord `python train_pipeline.py`.
- Les composants CrewAI/Ollama utilisent le modele `llama3.2` quand il est disponible.
