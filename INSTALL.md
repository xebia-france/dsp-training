# Installation du projet

Note : Cette procédure devrait fonctionner sur Linux et Mac. Il faut la tester et la compléter pour Windows.

## Dépendances

Le projet utilise `Python3` et des librairies listées dans le fichier [requirements](requirements.txt). 

Nous recommendons d'installer ces dépendances dans un environnement virtuel.

Tester l'installation de `Python3` et `virtualenv` avec ces commandes

    python3 --version
    pip3 --version
    virtualenv --version
    
Note: selon votre installation de python, vous pouvez remplacer les commandes `python3` et `pip3` par `python` et `pip`

Installer `virtualenv` si besoin avec la commande :  `pip3 install virtualenv`

## Installation 

Cloner le projet en local

    git clone https://gitlab.com/USER/PROJECT.git
    cd PROJECT

Créer un environnement virtuel et l'activer

    virtualenv venv --python=python3
    source venv/bin/activate
    
Note: pour les utilisateurs de Windows, la commande d'activation de l'environnement virtuel est
    
    venv\Scripts\activate.bat

Installer les dépendances python 

    pip3 install -r requirements.txt

Tester l'installation

    python -m pytest tests
