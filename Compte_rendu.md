# Compte-rendu — TP1 : Détection de Fake News avec un PLM

**Nom et prénom :** Oumayma Zerrik  
**Dataset choisi :** Option A — Fake and Real News Dataset (ISOT)  
**PLM utilisé :** DistilBERT (`distilbert-base-uncased`)

---

## 1. Résultats

| Approche | Accuracy | F1-score | Temps | GPU / Coût |
|---|---:|---:|---:|---|
| A — TF-IDF + Logistic Regression | 0.987973 | 0.987454 | 89.42 s | CPU / Pas de GPU |
| B — DistilBERT gelé + Logistic Regression | 0.993987 | 0.993699 | 233.10 s | GPU T4 |
| C — DistilBERT Fine-tuning | 0.999555 | 0.999533 | 369.80 s (~6.16 min) | GPU T4 |
| Bonus — LLM prompté | Non réalisé | Non réalisé | — | — |

---

## 2. Matrice de confusion

### Matrice de confusion — Baseline TF-IDF + Logistic Regression

> **Insérer ici la capture de la matrice de confusion de la baseline**

```text
[ CAPTURE MATRICE DE CONFUSION TF-IDF + LR ]
```

### Matrice de confusion — DistilBERT gelé + Logistic Regression

> **Insérer ici la capture de la matrice de confusion du PLM gelé**

```text
[ CAPTURE MATRICE DE CONFUSION DISTILBERT GELÉ + LR ]
```

### Matrice de confusion — DistilBERT Fine-tuning

> **Insérer ici la capture de la matrice de confusion du modèle fine-tuné**

```text
[ CAPTURE MATRICE DE CONFUSION FINE-TUNING ]
```

---

# 3. Questions

## Q1 — Les classes sont-elles équilibrées ? Quel impact sur le choix de la métrique ?

Les classes du dataset ISOT sont relativement équilibrées.

L'Accuracy est donc une métrique pertinente pour mesurer les performances globales.  
Nous utilisons également le F1-score, car il combine la précision et le rappel et permet d'avoir une évaluation plus complète de la classification.

---

## Q2 — Quel score obtient la baseline TF-IDF + Logistic Regression ?

La baseline **TF-IDF + Logistic Regression** obtient :

- **Accuracy : 98.80 %**
- **F1-score : 98.75 %**
- **Temps : 89.42 secondes**

Ces résultats montrent qu'une approche classique et relativement simple permet déjà d'obtenir de très bonnes performances sur le dataset ISOT.

---

## Q3 — PLM gelé vs baseline : quel gain ? Pourquoi les embeddings contextuels aident-ils ?

La baseline obtient un F1-score de :

```text
98.75 %
```

DistilBERT gelé + Logistic Regression obtient :

```text
99.37 %
```

Le gain est donc d'environ :

```text
99.37 - 98.75 = +0.62 point de pourcentage
```

Cette amélioration peut être expliquée par les embeddings contextuels de DistilBERT.

TF-IDF représente principalement l'importance statistique des mots, tandis que DistilBERT prend en compte le contexte dans lequel les mots apparaissent.

Dans cette approche, DistilBERT est utilisé comme extracteur de features sans modifier ses poids. Les embeddings produits sont ensuite utilisés par une Logistic Regression pour effectuer la classification.

---

## Q4 — Fine-tuning vs PLM gelé : quel gain ? À quel coût ?

DistilBERT gelé obtient :

```text
F1 = 99.37 %
Temps = 233.10 secondes
```

DistilBERT fine-tuné obtient :

```text
F1 = 99.95 %
Temps = 369.80 secondes (~6.16 minutes)
```

Le gain est donc d'environ :

```text
99.95 - 99.37 = +0.58 point de pourcentage
```

Le fine-tuning améliore les performances, mais nécessite davantage de temps de calcul et l'utilisation d'un GPU T4.

Dans l'approche gelée, les poids de DistilBERT ne sont pas modifiés.

Dans le fine-tuning, les poids de DistilBERT ainsi que la tête de classification sont ajustés pour la tâche Fake/Real.

---

## Q5 — Matrice de confusion : quelles classes se confondent ? Donnez 2 exemples d'erreurs et une hypothèse.

Les matrices de confusion montrent que très peu d'articles sont mal classés, ce qui est cohérent avec les F1-scores très élevés obtenus.

Les erreurs peuvent concerner des articles dont le vocabulaire ou le style rédactionnel ressemble fortement à celui de l'autre classe.

### Exemple d'erreur 1

> À compléter avec un vrai article mal classé obtenu dans le notebook.

**Classe réelle :** ______  
**Classe prédite :** ______  

**Texte / extrait :**

```text
À insérer ici.
```

### Exemple d'erreur 2

> À compléter avec un deuxième article mal classé obtenu dans le notebook.

**Classe réelle :** ______  
**Classe prédite :** ______  

**Texte / extrait :**

```text
À insérer ici.
```

### Hypothèse

Certains articles Fake peuvent adopter un style journalistique très proche des articles Real. Inversement, certains articles Real peuvent contenir des formulations atypiques qui ressemblent à celles présentes dans les Fake News.

---

## Q6 — Quel modèle choisiriez-vous pour la production, et pourquoi ?

Les résultats montrent le compromis suivant :

| Modèle | F1-score | Temps | GPU |
|---|---:|---:|---|
| TF-IDF + LR | 98.75 % | 89.42 s | Non |
| DistilBERT gelé + LR | 99.37 % | 233.10 s | T4 |
| DistilBERT Fine-tuning | 99.95 % | 369.80 s | T4 |

La baseline TF-IDF + Logistic Regression est intéressante lorsqu'on privilégie la simplicité, la rapidité et un faible coût de calcul.

Le fine-tuning de DistilBERT fournit la meilleure performance obtenue dans nos expériences, mais nécessite davantage de ressources.

Le choix pour la production dépend donc du compromis entre **qualité, coût et latence**. Une évaluation sur de nouvelles sources et une mesure de la latence d'inférence seraient nécessaires avant un déploiement réel.

---

## Q7 — Comment avez-vous évité le sur-apprentissage et la fuite test-in-train ?

Les données ont été séparées en trois ensembles indépendants :

- **80 % Train**
- **10 % Validation**
- **10 % Test**

Une seed fixe a été utilisée :

```python
SEED = 42
```

La séparation est également stratifiée afin de conserver la distribution des classes.

Pour TF-IDF, `fit_transform()` est appliqué uniquement sur le Train :

```python
X_train_tfidf = tfidf.fit_transform(X_train)
```

Pour Validation et Test, nous utilisons uniquement :

```python
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)
```

Cela permet d'éviter que des informations provenant du jeu de test soient utilisées pendant l'entraînement.

Pour le fine-tuning, la Validation Loss a été surveillée :

| Epoch | Training Loss | Validation Loss | F1 |
|---:|---:|---:|---:|
| 1 | 0.005487 | 0.001630 | 0.999533 |
| 2 | 0.002022 | 0.000930 | 0.999767 |
| 3 | 0.000068 | 0.000697 | 0.999767 |

La Validation Loss continue de diminuer et le F1 reste stable à un niveau très élevé. Aucun signe classique de sur-apprentissage n'a donc été observé pendant les trois epochs.

---

## Q8 — Bonus : un LLM bien prompté fait-il mieux ou moins cher que votre PLM fine-tuné ?

Cette partie bonus n'a pas été réalisée.

Il n'est donc pas possible de conclure expérimentalement si un LLM utilisé en zero-shot ou few-shot serait plus performant ou moins coûteux que DistilBERT fine-tuné.

---

# 4. Décision & justification

La baseline **TF-IDF + Logistic Regression** obtient déjà un excellent F1-score de **98.75 %** avec le temps de calcul le plus faible et sans GPU.

DistilBERT gelé améliore le F1-score à **99.37 %**, tandis que le fine-tuning atteint **99.95 %**.

Le fine-tuning fournit donc la meilleure performance mesurée, tandis que TF-IDF + Logistic Regression est l'approche la plus simple et la moins coûteuse.

Le choix final dépend des contraintes de production : performances recherchées, ressources disponibles et latence acceptable.

---

# 5. Limites & pistes d'amélioration

Une limite importante du dataset ISOT est que le **style rédactionnel ou la source des articles peuvent donner des indices sur leur classe**.

Les performances très élevées doivent donc être interprétées avec prudence.

Un F1-score de **99.95 % sur ISOT ne garantit pas la même performance sur de nouvelles sources**.

Les principales pistes d'amélioration sont :

- tester le modèle sur des articles provenant de nouvelles sources ;
- analyser davantage les erreurs de classification ;
- vérifier l'influence de la source et du style rédactionnel ;
- tester d'autres PLM ;
- mesurer la latence réelle d'inférence en production.
