# Machine Learning Challenge Report
## Challenge 0: Il Problema dell'Autenticazione delle Banconote
## *Dino Meng, SM3201466*

---

**SCALETTA**

* Introduzione: introduco il problema, dataset eccetera...
- Traduco l'introduzione nel notebook

* Metodologia: descrivo brevemente la pipeline per il data preprocessing
- EDA: Nessun valore mancante, alcune variabili sono distorte e non sono nella stessa scala 
- Variabili standardizzate con StandardScaler
- I dati sono stati mescolati in quanto erano organizzati secondo le classi (target variable)

* Unsupervised Learning
- PCA: Uso PCA per ridurre la dimensionalità del problema (da quattro a tre/due)
- T-SNE: Il problema diventa linearmente separabile (tranne per alcuni punti singolari)
- Uso KMeans DBSCAN e altre tecniche per effettuare clustering sul dataset trasformato, per:
    - Trovare punti di rumore
    - Riprodurre in una maniera fedele le classi originarie

* Supervised Learning
- Ho usato di nuovo il metodo hold out, col test size = 372
- Ho fittato più modelli al dataset originario (scalato), ossia:
    - Albero decisionale ID3 (Usa la loss in entropia)
    - Regressioni logistiche (regolarizzate e non)
    - Naive Bayes Gaussiano (le variabili non sono categoriche, pertanto usiamo la stima gaussiana)
    - KNN
- Di nuovo ma usando la trasformazione fatta con T-SNE

* Risultati: 
- Unsupervised Learning:
    - PCA non è riuscito a trasformare il dataset in un problema separabile (quindi neanche linearmente separabile)
    - Invece T-SNE è riuscito a trasformare il dataset in un problema linearmente separabile
    - nè KMeans nè DBSCAN riescono a riprodurre le classi originarie 
    - Hierarchical Clustering ci riesce sul dataset T-SNE

- Supervised Learning:
Riporto il grafico delle F1 score

* Discussione: discuto i risultati 
Unsupervised learning: 
- ???

Supervised Learning
Il miglior modello era KNN col test F1-score di 1.0
Il peggior modello era Naive Bayes (probabilmente il dataset non era separabile, per cui fallisce l'assunzione dell'indipendenza tra le variabili)
La regressione logistica aveva una buona performance, meglio se non subisce nessuna forma di regolarizzazione

Un modo per migliorare i risultati è quello di usare il dataset trasformato con T-SNE: infatti facendolo si ottiene un boost significativo delle performance dei modelli (quasi tutti i modelli superavano 0.99 di F1-test score), comunque i risultati sono comparabili a quelli di prima

TO DO: provare con PCA (?)


> `pandoc .\report.md -o .\report.pdf -V geometry:margin=1in` to export document