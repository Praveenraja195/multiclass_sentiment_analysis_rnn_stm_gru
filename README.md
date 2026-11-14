# Multiclass Sentiment Analysis: Comparative Deep Learning & Automated Hyperparameter Tuning

An end-to-end NLP sentiment classification pipeline comparing **Dense Neural Networks (MLP)**, **SimpleRNN**, **LSTM**, and **GRU** architectures on the [`Sp1786/multiclass-sentiment-analysis-dataset`](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset) (41.6k samples across **Negative**, **Neutral**, and **Positive** classes), featuring automated Bayesian hyperparameter optimization via **Keras Tuner**.

---

## Project Overview & Pipeline

1. **Data Ingestion & Cleaning:** Streams 41,600 labeled samples (`31.2k` train / `5.21k` validation / `5.21k` test) directly from Hugging Face, applying regex normalization (stripping HTML tags, URLs, and `@mentions` while preserving negations and punctuation) and leakage-free `TextVectorization` (`MAX_VOCAB = 15,000`, `MAX_LEN = 100`).
2. **Unified Architectural Comparison:** Evaluates four distinct neural architectures—Feedforward **Dense NN**, **SimpleRNN**, **LSTM**, and **GRU**—on identical token representations.
3. **Automated Hyperparameter Tuning:** Uses `keras_tuner.BayesianOptimization` with `EarlyStopping` to automatically optimize embedding dimensions, recurrent/dense units, bidirectionality, dropout rates, and learning rates.
4. **Diagnostics & Unseen Evaluation:** Logs validation accuracy, Macro F1-score, training/validation convergence curves, confusion matrix heatmaps, and live inference predictions.

---

## Model Comparison Leaderboard

| Rank | Architecture | Val Accuracy | Val Macro F1 | Trainable Params | Optimal Hyperparameters |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Bidirectional GRU** | **68.47%** | **0.6876** | 1,109,763 | `embed_dim=64`, `gru_units=128`, `bidirectional=True`, `dropout=0.5`, `lr=0.001` |
| **2** | **Bidirectional LSTM** | 68.18% | 0.6848 | 1,158,403 | `embed_dim=64`, `lstm_units=128`, `bidirectional=True`, `dropout=0.5`, `lr=0.001` |
| **3** | **SimpleRNN** | 65.80% | 0.6612 | 1,986,563 | `embed_dim=128`, `rnn_units=128`, `bidirectional=True` |
| **4** | **Dense NN (MLP)** | 62.98% | 0.6239 | 968,515 | `embed_dim=64`, `num_layers=2` |

### Key Architectural Takeaways
* **Why Gated Bidirectional Models Won:** Unlike the Dense NN baseline (which averages sequence embeddings via `GlobalAveragePooling1D`), **Bidirectional GRU** and **Bidirectional LSTM** process text both left-to-right and right-to-left, preserving word order and linking negations/modifiers (*"not very good"*, *"crashes constantly"*) directly to sentiment cues (`+5.5%` accuracy gain over Dense NN).
* **Parameter Efficiency (`GRU` vs. `LSTM` & `SimpleRNN`):** Bidirectional GRU achieved the top validation performance with **~48.6k fewer parameters** than LSTM and **~877k fewer parameters** than SimpleRNN, avoiding vanishing gradients while generalizing cleanly on 31.2k training samples.

---

## Tech Stack
* **Deep Learning & Tuning:** TensorFlow / Keras, Keras Tuner (`BayesianOptimization`)
* **Data & Metrics:** Hugging Face `datasets`, NumPy, Pandas, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

---

## How to Run
1. Open [`HclTech_ass2.ipynb`](./HclTech_ass2.ipynb) in Google Colab.
2. Enable GPU acceleration (**Runtime → Change runtime type → T4 GPU**).
3. Run all cells sequentially (**Runtime → Run all**) to reproduce the EDA plots, tuning trials
