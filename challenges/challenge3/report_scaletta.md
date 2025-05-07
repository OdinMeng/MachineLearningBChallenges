SCALETTA DEL REPORT

* Introduzione: introduco il dataset, lo scopo del report (sperimentare tecniche del deep learning, sia con NN feed-forward che NN convoluzionali)

* Metodologia:
    * EDA: Visualizzo i dati per avere l'idea del dataset
    * Preparazione dei dati: Solamente splittato il dataset del training in training e testing col metodo del hold out, seguendo la proporzione 70-30. Inoltre i dati sono stati rimescolati, per garantire che la distribuzione delle classi siano veramente casuali. Il dataset del testing è stato fornito a parte.
    * Architettura delle NN:
        * 2 tipi di CNN: una "shallow", contenente un layer convoluzionale, un layer di maxpooling e un layer lineare; l'altra "deep" che contiene più layer convoluzionali (3) e il layer lineare come quella finale
        * FCNN: analogamente, una deep con tre hidden layer e una shallow con una hidden layer
    * Training di un modello:
        * Useremo Adam e SGD come ottimizzatori
        * Passeremo dei batch da 128 immagini, data la dimensione del dataset
        * Inoltre useremo dei scheduler per modificare il learning rate di passo in passo. In particolare, sarà un scheduler ???, che modifica a seconda della variazione della test loss.
    * Valutazione di un modello:
        * Per valutare un modello abbiamo:
            * La micro accuracy, che va semplicemente a calcolare la percentuale dei dati correttamente classificati
            * Matrice della confusione multiclasse, ovvero una matrice M \in R^10 \times 10 dove M[i, j] è uguale alla quantità di oggetti di classe i classificati come j
            * Dalla CM derivano la macro precisione, macro recall e macro f1-score (facendo la media di queste per ogni classe)

* Risultati
    * Merda totale, regolarizzazione totalmente inutile. Il fatto che le FCNN hanno dominato in termini di performance è un'anomalia, boh non so io mi sparo.

* Discussione
    * Generalmente c'è overfit su tutti i modelli
    * Specificamente, studiamo la CNN allenata con Adam notiamo che:
        * è intervenuto lo scheduler varie volte, è associato con i cali di loss (più visibilmente nel primo intervento)
        * creo la confusione di matrice e calcolo precision e recall; sono quasi uguali, quindi possiamo dire che la precision va bene lo stesso per il nostro problema
        * infatti, guardando gli istogrammi si differiscono al più di circa 0.0125
        * le classi con cui ha meno confidenza sono 2, 4, 6

* Concolusione: ho merda totale. cosa posso fare???
    * metodi per mitigare l'overfitting:
        * dropout, dropblock
        * arresto anticipato, il numero di iterazioni erano piuttosto alte e causavano overfit