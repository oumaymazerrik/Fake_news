# TP1 — Détection de Fake News avec un PLM

## Compte-rendu

**Nom et prénom :** Oumayma Zerrik  
**Filière / Groupe :** À compléter  
**Date :** À compléter  
**Dataset choisi :** Option A — Fake and Real News Dataset (ISOT)  
**PLM utilisé :** `distilbert-base-uncased`

---

# 1. Objectif

L'objectif de ce TP est de détecter automatiquement les **Fake News** à partir de textes en anglais en utilisant différentes approches de classification.

Trois approches ont été comparées :

1. **Baseline : TF-IDF + Logistic Regression**
2. **PLM gelé : DistilBERT + Logistic Regression**
3. **Fine-tuning : DistilBERT pour la classification**

L'objectif est de comparer les modèles selon :

- Accuracy
- Precision
- Recall
- F1-score
- Temps d'entraînement / traitement
- Coût de calcul
- Besoin ou non d'un GPU

---

# 2. Dataset utilisé

Le dataset choisi est :

**Fake and Real News Dataset (ISOT)**

Source Kaggle :

```text
clmentbisaillon/fake-and-real-news-dataset
```

Le dataset contient environ **44 000 articles en anglais** répartis dans deux fichiers :

```text
Fake.csv
True.csv
```

Les labels utilisés sont :

```text
0 = Fake
1 = Real
```

Le contenu utilisé pour la classification est construit à partir du titre et du texte :

```python
df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)
```

---

# 3. Séparation des données

Le même découpage est utilisé pour les trois approches :

```text
Dataset
   │
   ├── 80 % Train
   ├── 10 % Validation
   └── 10 % Test
```

Une seed fixe est utilisée afin d'améliorer la reproductibilité :

```python
SEED = 42
```

La séparation est stratifiée afin de conserver approximativement la même distribution des classes dans les différents ensembles.

---

# 4. Approche A — TF-IDF + Logistic Regression

La première approche constitue notre **baseline**.

Le texte est transformé en représentation numérique avec **TF-IDF**, puis les vecteurs obtenus sont fournis à une **régression logistique**.

Architecture :

```text
Article
   ↓
TF-IDF
   ↓
Vecteur
   ↓
Logistic Regression
   ↓
Fake / Real
```

Cette approche ne nécessite pas de PLM ni de GPU.

---

# 5. Approche B — DistilBERT gelé + Logistic Regression

Le modèle utilisé est :

```text
distilbert-base-uncased
```

DistilBERT est utilisé comme **extracteur de features**.

Ses paramètres sont gelés :

```python
for param in plm_model.parameters():
    param.requires_grad = False
```

Cela signifie que les poids de DistilBERT ne sont pas modifiés.

Architecture :

```text
Article
   ↓
Tokenizer
   ↓
DistilBERT gelé
   ↓
Embedding
   ↓
Logistic Regression
   ↓
Fake / Real
```

Dans cette approche :

- **DistilBERT = extracteur de features**
- **Logistic Regression = classifier**

---

# 6. Approche C — Fine-tuning de DistilBERT

Pour la troisième approche, le modèle utilisé est également :

```text
distilbert-base-uncased
```

Mais cette fois, DistilBERT est chargé avec :

```python
AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
```

`num_labels=2` correspond aux deux classes :

```text
0 → Fake
1 → Real
```

Architecture :

```text
Article
   ↓
Tokenizer
   ↓
DistilBERT
   ↓
Classification Head
   ↓
2 logits
   ↓
Fake / Real
```

Contrairement à l'approche précédente, les poids du modèle sont ajustés pendant l'entraînement.

Les principaux paramètres du fine-tuning sont :

```text
Epochs                  : 3
Learning rate           : 2e-5
Train batch size        : 32
Evaluation batch size   : 64
Maximum sequence length : 128
Weight decay            : 0.01
Seed                    : 42
GPU                     : Tesla T4
```

---

# 7. Résultats

Les résultats obtenus sont :

| Approche | Accuracy | F1-score | Temps | GPU |
|---|---:|---:|---:|---|
| TF-IDF + Logistic Regression | 0.987973 | 0.987454 | 89.42 s | Non |
| DistilBERT gelé + Logistic Regression | 0.993987 | 0.993699 | 233.10 s | T4 |
| DistilBERT Fine-tuning | 0.999555 | 0.999533 | 369.80 s | T4 |

En pourcentage :

| Approche | Accuracy | F1-score |
|---|---:|---:|
| TF-IDF + LR | 98.80 % | 98.75 % |
| DistilBERT gelé + LR | 99.40 % | 99.37 % |
| DistilBERT Fine-tuning | 99.96 % | 99.95 % |

---

# 8. Résultats du Fine-tuning

Les performances obtenues pendant les trois epochs sont :

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.005487 | 0.001630 | 0.999555 | 1.000000 | 0.999066 | 0.999533 |
| 2 | 0.002022 | 0.000930 | 0.999777 | 0.999533 | 1.000000 | 0.999767 |
| 3 | 0.000068 | 0.000697 | 0.999777 | 0.999533 | 1.000000 | 0.999767 |

Temps total du fine-tuning :

```text
≈ 369.8 secondes
≈ 6.16 minutes
```

---

# 9. Questions

## Q1 — Les classes sont-elles équilibrées ? Quel impact sur le choix de la métrique ?

Les deux classes du dataset ISOT sont relativement équilibrées.

L'**Accuracy** est donc une métrique pertinente pour évaluer les performances globales.

Cependant, nous utilisons également le **F1-score**, car il combine :

```text
Precision + Recall
```

et permet de mieux analyser les performances de classification.

---

## Q2 — Quel score obtient la baseline TF-IDF + Logistic Regression ?

La baseline obtient :

```text
Accuracy = 0.987973 ≈ 98.80 %
F1       = 0.987454 ≈ 98.75 %
```

Ces résultats montrent qu'une méthode classique relativement simple donne déjà d'excellentes performances sur le dataset ISOT.

---

## Q3 — PLM gelé vs baseline : quel gain ?

La baseline obtient :

```text
F1 = 98.75 %
```

Le PLM gelé obtient :

```text
F1 = 99.37 %
```

Le gain est donc d'environ :

```text
99.37 - 98.75 ≈ +0.62 point de pourcentage
```

### Pourquoi DistilBERT peut-il améliorer les résultats ?

TF-IDF représente principalement l'importance statistique des mots.

DistilBERT produit des **embeddings contextuels**.

Cela signifie que la représentation d'un mot dépend des autres mots qui l'entourent.

Par exemple, le sens d'un même mot peut changer selon son contexte.

Dans notre approche :

```text
DistilBERT gelé
      ↓
Embedding
      ↓
Logistic Regression
```

DistilBERT fournit donc une représentation sémantique du texte avant la classification.

---

## Q4 — Fine-tuning vs PLM gelé : quel gain et à quel coût ?

Le PLM gelé obtient :

```text
F1 = 99.37 %
```

Le modèle fine-tuné obtient :

```text
F1 = 99.95 %
```

Le gain est donc d'environ :

```text
99.95 - 99.37 ≈ +0.58 point de pourcentage
```

Cependant, le coût de calcul augmente.

```text
PLM gelé :
233.10 secondes
≈ 3.9 minutes

Fine-tuning :
369.80 secondes
≈ 6.16 minutes
```

Le fine-tuning nécessite également un GPU dans notre expérience.

La différence principale est :

```text
PLM gelé
→ poids DistilBERT non modifiés

Fine-tuning
→ poids DistilBERT ajustés pour la tâche Fake/Real
```

---

## Q5 — Matrice de confusion et analyse des erreurs

Les performances obtenues sont très élevées, ce qui signifie que très peu d'articles sont mal classés.

Les erreurs peuvent concerner des articles dont le vocabulaire ou le style rédactionnel ressemble fortement à celui de l'autre classe.

### Exemple d'erreur 1

```text
À compléter avec un véritable exemple mal classé obtenu dans le notebook.
```

### Exemple d'erreur 2

```text
À compléter avec un véritable exemple mal classé obtenu dans le notebook.
```

### Hypothèse

Certains articles Fake peuvent adopter un style journalistique très proche des articles Real.

Inversement, certains articles Real peuvent contenir des formulations ou un vocabulaire atypiques.

Cela peut rendre leur classification plus difficile.

---

## Q6 — Quel modèle choisir pour la production ?

Les trois modèles présentent différents compromis.

### TF-IDF + Logistic Regression

```text
F1 : 98.75 %
Temps : 89.42 s
GPU : Non
```

Avantages :

- simple
- rapide
- faible coût
- très bonnes performances

### DistilBERT gelé + LR

```text
F1 : 99.37 %
Temps : 233.10 s
GPU : T4
```

Il améliore les performances grâce aux embeddings contextuels, mais demande davantage de calcul.

### DistilBERT Fine-tuning

```text
F1 : 99.95 %
Temps : 369.80 s
GPU : T4
```

Il fournit les meilleures performances obtenues dans cette expérience, mais avec un coût de calcul supérieur.

Le choix en production dépend donc du compromis recherché entre :

```text
Qualité
   ↕
Coût
   ↕
Latence
```

Avant un déploiement réel, il serait nécessaire de mesurer la latence d'inférence et surtout de tester les modèles sur des données provenant de nouvelles sources.

---

## Q7 — Comment éviter le sur-apprentissage et la fuite de données ?

Les données ont été séparées en trois ensembles indépendants :

```text
80 % Train
10 % Validation
10 % Test
```

La seed suivante a été utilisée :

```python
SEED = 42
```

La stratification permet également de conserver la distribution des classes.

### Pour TF-IDF

TF-IDF est entraîné uniquement sur le Train :

```python
X_train_tfidf = tfidf.fit_transform(X_train)
```

Pour Validation et Test :

```python
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)
```

On ne fait donc pas de :

```python
fit_transform(X_test)
```

Cela évite d'utiliser des informations provenant du jeu de test pendant l'apprentissage.

### Pour le Fine-tuning

Les performances de validation ont été surveillées pendant les epochs.

Validation Loss :

```text
Epoch 1 → 0.001630
Epoch 2 → 0.000930
Epoch 3 → 0.000697
```

La Validation Loss continue donc de diminuer.

Le F1 de validation reste également très élevé.

Aucun signe classique de sur-apprentissage n'a été observé pendant les trois epochs réalisés.

---

## Q8 — Bonus : comparaison avec un LLM prompté

Cette partie bonus n'a pas été réalisée.

Il n'est donc pas possible de conclure expérimentalement si un LLM utilisé en **zero-shot** ou **few-shot** serait plus performant ou moins coûteux que notre DistilBERT fine-tuné.

---

# 10. Comparaison finale

```text
TF-IDF + LR
     │
     │ F1 = 98.75 %
     ▼
DistilBERT gelé + LR
     │
     │ F1 = 99.37 %
     ▼
DistilBERT Fine-tuning
     │
     │ F1 = 99.95 %
     ▼
Meilleure performance obtenue
```

Le passage de la baseline au PLM gelé apporte environ :

```text
+0.62 point de F1
```

Le passage du PLM gelé au fine-tuning apporte environ :

```text
+0.58 point de F1
```

Le gain total entre la baseline et le fine-tuning est donc d'environ :

```text
+1.20 point de F1
```

---

# 11. Décision et justification

Les trois approches obtiennent des performances élevées sur le dataset ISOT.

La baseline **TF-IDF + Logistic Regression** atteint déjà un F1-score de **98.75 %**, avec le temps de calcul le plus faible et sans GPU.

Le modèle **DistilBERT gelé + Logistic Regression** atteint **99.37 %** de F1, mais nécessite davantage de calcul.

Le **fine-tuning de DistilBERT** atteint **99.95 %** de F1, avec le temps de calcul le plus important parmi les trois approches testées.

Le choix final en production doit donc dépendre des contraintes de l'application, notamment :

- performance recherchée ;
- ressources matérielles disponibles ;
- coût ;
- latence d'inférence ;
- capacité de généralisation.

Une validation sur des données externes serait nécessaire avant un véritable déploiement.

---

# 12. Limites

Une limite importante concerne le dataset ISOT.

Le style rédactionnel ou les sources des articles peuvent fournir des indices permettant au modèle de distinguer facilement les classes Fake et Real.

Cela peut contribuer aux performances extrêmement élevées obtenues.

Par conséquent :

```text
99.95 % sur ISOT
        ≠
99.95 % sur toutes les nouvelles Fake News
```

Les résultats doivent donc être interprétés avec prudence.

---

# 13. Pistes d'amélioration

Plusieurs améliorations peuvent être envisagées :

- tester le modèle sur des articles provenant de nouvelles sources ;
- réaliser une analyse plus détaillée des erreurs ;
- vérifier l'existence d'indices liés aux sources dans le dataset ;
- tester d'autres PLM ;
- comparer différentes longueurs maximales de séquence ;
- mesurer la latence d'inférence ;
- réaliser le bonus avec un LLM en zero-shot ou few-shot.

---

# 14. Reproductibilité

Le projet contient :

```text
Fake_news/
│
├── TP1_Fake_News.ipynb
├── requirements.txt
├── README.md
└── COMPTE_RENDU.md
```

Le fichier `requirements.txt` contient les principales dépendances :

```text
numpy
pandas
matplotlib
scikit-learn
torch
transformers
datasets
accelerate
evaluate
kagglehub
tqdm
```

Installation :

```bash
pip install -r requirements.txt
```

Le notebook peut être exécuté avec **Google Colab**.

Pour les expériences avec DistilBERT, un **GPU T4** est recommandé.

La seed utilisée est :

```python
SEED = 42
```

Cela permet d'améliorer la reproductibilité des expériences.
  
