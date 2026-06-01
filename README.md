# Bachelorarbeit: Vergleichende Untersuchung hierarchischer Textklassifikationsstrategien

Dieses Repository enthält den vollständigen Quellcode zur Bachelorarbeit "Vergleichende Untersuchung hierarchischer Textklassifikationsstrategien: Diskriminative Ansätze im Vergleich über verschiedene Taxonomien". Es implementiert eine Trainings- und Evaluationspipeline zum systematischen Vergleich von fünf hierarchischen Klassifikationsparadigmen (Flat Baseline, LCL, LCPN, GMH, GCC) basierend auf einem DistilBERT-Encoder.

## 1. Voraussetzungen und Installation

Die Experimente und Modelle wurden in der folgenden Software-Umgebung entwickelt und getestet:
* **Python:** 3.12.13
* **PyTorch:** 2.10.0 (+ CUDA)
* **Transformers (Hugging Face):** 5.0.0
* **Optuna:** 4.8.0
* **Scikit-learn:** 1.6.1

Zur Ausführung empfiehlt sich die Erstellung einer virtuellen Umgebung. Die Abhängigkeiten können (sofern eine `requirements.txt` vorhanden ist) wie folgt installiert werden:
```bash
pip install -r requirements.txt
```

## 2. Datensätze

Die Pipeline verarbeitet drei unterschiedliche Evaluierungsdatensätze:
* **WOS und DBpedia:** Diese beiden Datensätze werden bei Ausführung automatisch über die Hugging Face `datasets` Bibliothek geladen.
* **GPSD:** Der im Rahmen der Arbeit generierte synthetische Google Product Synthetic Dataset (GPSD) ist in diesem Repository lokal als CSV-Datei hinterlegt und wird direkt über das Dateisystem eingelesen.

## 3. Projektstruktur

Die Architektur des Repositories gliedert sich in folgende Hauptverzeichnisse:

* `src/`: Enthält den vollständigen Python-Quellcode.
  * `data/`: Datenvorverarbeitung (`dataloader.py`, `dataset.py`, `hierarchy.py`).
  * `models/`: Implementierung der spezifischen Klassifikationsparadigmen (`baseline_flat.py`, `global_chaining.py`, `global_multi.py`, `local_lcl.py`, `local_lcpn.py`).
  * `training/`: Zentrale Trainingslogik (`trainer.py`, `loss.py`, `metrics.py`).
  * `utils/`: Hilfsfunktionen und Konfiguration (`logger.py`, `config.py`).
* `data/` (Root): Dient als Cache und beinhaltet vorberechnete bzw. gespeicherte Hierarchie-Informationen der Datensätze sowie die lokale CSV-Datei für den GPSD.
* `logs/`: Speichert die Log-Dateien aller Ergebnisse und Zwischenauswertungen.
* `models_saved/`: Beinhaltet alle lokal gespeicherten Modell-Checkpoints und final trainierten Modelle.

## 4. Ausführung der Experimente

Die Steuerung der Pipeline erfolgt primär über die Kommandozeile mittels zweier zentraler Skripte im Root- bzw. `src`-Verzeichnis:

### Standard-Training (Basis-Experimente)
Über `train_experiment.py` werden die regulären Trainingsdurchläufe gestartet. Hierüber lassen sich das Architektur-Paradigma (`--model_type`), der Datensatz (`--dataset`), die Fine-Tuning-Strategie (`--freeze_n_layers`) sowie der Name des Experiments definieren. 

Beispielhafter Aufruf für die Flat Baseline auf dem WOS-Datensatz mit vollständigem Fine-Tuning (0 eingefrorene Schichten):
```bash
python src/train_experiment.py --model_type flat --dataset wos --freeze_n_layers 0 --experiment_name "wos_flat_freeze_0" --learning_rate 2e-5
```

Alle Hyperparameter können zur Reproduktion der finalen Ergebnisse explizit über Kommandozeilen-Argumente überschrieben werden. Ein Aufruf mit den durch Optuna ermittelten optimalen Parametern sieht beispielsweise so aus:
```bash
python src/train_experiment.py --model_type flat --dataset wos --freeze_n_layers 0 --experiment_name "wos_flat_optimal_freeze_0" --learning_rate 1.8465e-05 --batch_size 32 --weight_decay 0.0910 --warmup_steps 500
```

### Hyperparameter-Optimierung (Optuna)
Das Skript `hyperparam_search.py` startet die Bayes'sche Hyperparameter-Optimierung (TPE-Sampler mit Median-Pruner) für eine spezifische Architektur-Datensatz-Kombination.

Beispielhafter Aufruf für den Suchprozess:
```bash
python src/hyperparam_search.py --model_type flat --dataset wos
```
