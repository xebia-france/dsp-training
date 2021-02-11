# dsp-training

code for dsp-training

Pour installer le projet, lire le fichier [INSTALL.md](INSTALL.md).

Ce projet a été créé en suivant ce [template cookiecutter de workflow pour un projet de data science en local](https://gitlab.com/VincentVillet/cookiecutter-data-fr).

Les propositions de bonnes pratiques suivantes sont implémentées dans le projet:

- centralisation de toutes les informations relatives aux dossiers et aux fichiers dans [src/constants/files.py](src/constants/files.py).
- variabilisation des features par dataframe dans [src/constants/columns.py](src/constants/columns.py). Cela permet de maintenir une documentation des features de façon transparente dans le code.
- centralisation des informations relatives aux modèles dans [src/constants/models.py](src/constants/models.py).
- "test d’intégration gratuit" dans [tests/free_integration_test](tests/integration_test). Plus d’informations sur ce test et son intérêt dans le [README de tests/free_integration_test](tests/integration_test/README.md).
- écriture des logs dans data/logs avec un fichier par jour.
- téléchargement automatique des données lors du lancement du [script main.py](main.py)
- Gestion du versionning des notebooks grâce à jupytext. Voir le [README du dossier notebooks](notebooks/README.md) pour plus d’information sur le fonctionnement de jupytext. La philosophie du projet est que les **notebooks servent exclusivement à faire de l’exploration de données au brouillon**. Tous les notebooks devraient pouvoir être supprimés à la fin du projet sans perdre aucune information utile.

Le template cookiecutter illustre ces pratiques à travers l’exemple simple d’une régression linéaire de la consommation d’énergie annuelle mondiale en fonction du PIB annuel mondial (inspirée d’un [article du blog de Jean-Marc Jeancovici](https://jancovici.com/transition-energetique/l-energie-et-nous/lenergie-de-quoi-sagit-il-exactement/)).