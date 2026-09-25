# TP1 — Fake News Detection with PLM

## Description

Ce projet a pour objectif de détecter automatiquement les fake news à partir de textes en anglais.

Trois approches sont comparées :

1. Baseline : TF-IDF + Logistic Regression
2. PLM gelé : DistilBERT comme extracteur de features + Logistic Regression
3. Fine-tuning : DistilBERT fine-tuné pour la classification binaire

---

## Dataset

Dataset utilisé : **Fake and Real News Dataset (ISOT)**

Source Kaggle :
`clmentbisaillon/fake-and-real-news-dataset`

Le dataset contient environ 44 000 articles en anglais répartis dans deux fichiers :

- `Fake.csv` : fausses informations
- `True.csv` : vraies informations

Labels utilisés :

- `0` = Fake
- `1` = Real

Le dataset est téléchargé automatiquement dans le notebook avec `kagglehub`.

---

## Structure du projet

```text
Fake_news/
│
├── TP1_Fake_News.ipynb
├── requirements.txt
└── README.md
