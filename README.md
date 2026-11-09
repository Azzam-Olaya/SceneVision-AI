# SceneVision AI — Classification de scènes

## Objectif

Classer une image dans l’une des six catégories du jeu Intel Image Classification : **buildings**, **forest**, **glacier**, **mountain**, **sea** ou **street**. Le projet utilise un CNN construit avec TensorFlow/Keras.

## Données et séparation

Le dossier `data/raw/seg_train` contient 17 034 images. Le prétraitement a trouvé six classes et aucune image illisible. Les images sont réparties de façon stratifiée dans les manifestes réutilisés par les notebooks.

| Classe | Train | Validation | Test |
|---|---:|---:|---:|
| buildings | 1 839 | 395 | 394 |
| forest | 1 921 | 412 | 412 |
| glacier | 2 070 | 443 | 444 |
| mountain | 2 126 | 455 | 456 |
| sea | 1 949 | 418 | 417 |
| street | 2 018 | 432 | 433 |
| **Total** | **11 923** | **2 555** | **2 556** |

Les images sont redimensionnées en 150 × 150 pixels, converties en RGB et normalisées entre 0 et 1. L’augmentation de données est appliquée au train seulement. Les CSV dans `data/split` sont la source utilisée par les notebooks.

## Modèle et entraînement

Le CNN comporte trois blocs `Conv2D` et `MaxPooling2D` (32, 64 puis 128 filtres), suivis de `Flatten`, `Dense(128)`, `Dropout` et d’une couche `Softmax` à six sorties. L’optimiseur est Adam et la perte `sparse_categorical_crossentropy`. L’entraînement utilise `EarlyStopping`, `ReduceLROnPlateau` et un checkpoint sauvegardé au meilleur `val_loss`.

Trois configurations ont été comparées sur la validation :

| Configuration | Époques | Loss validation | Accuracy validation | Temps mesuré |
|---|---:|---:|---:|---:|
| Référence — Dropout 0,4, Adam 0,001 | 20 | 0,3968 | 86,07 % | non enregistré |
| Dropout 0,5, Adam 0,001 | 16 | **0,3882** | **86,11 %** | 36 min 01 s |
| Dropout 0,4, Adam 0,0005 | 16 | 0,3969 | 85,68 % | 38 min 01 s |

La variante Dropout 0,5 a été sélectionnée par la plus faible loss de validation. Son avantage d’accuracy sur la référence est très faible (environ 0,04 point de pourcentage). La configuration avec un learning rate de 0,0005 n’a pas amélioré les résultats. Le tableau complet se trouve dans `results/model_comparison.csv`.

## Évaluation finale du modèle sélectionné

L’évaluation du modèle sélectionné sur 2 556 images de test donne une accuracy de **84,39 %**, une F1 macro de **84,45 %** et **399 erreurs**. Le modèle de référence avait auparavant obtenu 85,21 % d’accuracy sur ce même jeu de test. Le choix retenu sur validation n’a donc pas produit de gain mesuré sur le test ; l’amélioration de validation n’est pas une preuve d’une meilleure généralisation.

| Classe | Precision | Recall | F1-score |
|---|---:|---:|---:|
| buildings | 0,83 | 0,79 | 0,81 |
| forest | 0,95 | 0,97 | 0,96 |
| glacier | 0,83 | 0,80 | 0,81 |
| mountain | 0,81 | 0,77 | 0,79 |
| sea | 0,85 | 0,88 | 0,87 |
| street | 0,80 | 0,86 | 0,83 |

La classe **forest** est la mieux reconnue. **Mountain** a le F1-score le plus faible. Les confusions les plus nombreuses sont buildings prédit comme street (65 images), street prédit comme buildings (45), et les confusions glacier/mountain dans les deux sens (52 images chacune). Les métriques détaillées et la matrice sont enregistrées dans `results/selected_model_test_metrics.csv` et `results/selected_model_confusion_matrix.csv`.

## Surapprentissage et augmentation

Sur la configuration Dropout 0,5, la meilleure loss de validation apparaît à l’époque 13 (0,3882). Elle remonte ensuite jusqu’à l’arrêt à l’époque 16, tandis que l’accuracy d’entraînement continue à augmenter. Cela indique un début de surapprentissage tardif, limité par l’arrêt anticipé et le retour aux meilleurs poids.

Toutes les configurations comparées utilisent la même augmentation. Il n’y a pas de modèle témoin entraîné sans augmentation ; son effet propre n’a donc pas été mesuré.

## Prédire une image

Le notebook `notebooks/prediction.ipynb` charge `cnn_scenevision_selected.keras` s’il est présent, sinon le modèle de référence. La fonction `predict_image(image_path)` redimensionne l’image, convertit ses pixels en RGB normalisé, puis affiche la classe prédite et les scores des six classes. Elle est testée sur des images non étiquetées de `data/raw/seg_pred`.

Les scores Softmax ne sont pas des garanties de justesse ni des probabilités calibrées. Le modèle choisit toujours une des six catégories, même si l’image ne représente aucune de ces scènes.

## Limites et conditions d’utilisation

- Le modèle ne reconnaît que les six catégories vues pendant l’entraînement.
- Des scènes ambiguës, des images de mauvaise qualité ou des environnements différents du dataset peuvent entraîner des erreurs.
- Buildings/street et glacier/mountain sont les confusions principales observées.
- Les performances proviennent d’un seul découpage du dataset et ne garantissent pas le même résultat sur des images prises dans d’autres conditions.
- Le notebook d’exploration compte les classes et vérifie les formats/dimensions ; l’affichage d’une grille d’exemples reste à compléter.

Pour utiliser la prédiction, ouvrir `notebooks/prediction.ipynb`, choisir une image et exécuter les cellules dans l’ordre. Le modèle attend une image couleur RGB et applique le même redimensionnement et la même normalisation que pendant l’entraînement.


## Interface Streamlit

L’application `app.py` présente l’accuracy du modèle et permet d’importer une image pour voir sa classe prédite et les scores des six classes.

### Lancement

Installer les dépendances dans l’environnement Python choisi, puis lancer l’application depuis la racine du projet :

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Le modèle est chargé une seule fois et l’interface réutilise les métriques déjà enregistrées dans `results/`.
