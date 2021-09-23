import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump
import logging

import src.constants.columns as c


def logistic_reg_train(preprocessed_train_path, logistic_reg_model_path):
    """
    Read preprocessed train data, instantiate model, fit on train data and save model.

    :param preprocessed_train_path: path to preprocessed training data.
    :param logistic_reg_model_path: path to saved logistic reg model.

    :return: None
    """

    # TODO 4 : lire le fichier des données d’entraînement préprocessées qui ont été précédemment sauvegardées dans
    #  l'étape de preprocessing et passer le bon paramètre preprocessed_train_path à la fonction dans le script main.py.
    """
    L'import de pandas a déjà été réalisé en haut de ce fichier: import pandas as pd
    """
    train_df = None

    logistic_reg = LogisticRegression()
    
    logistic_reg.fit(
        train_df.drop(c.Loans.target(), axis=1).values,
        train_df[c.Loans.target()].values
    )

    logging.info("Saving model")
    # TODO 5 : sauvegarder le modèle logistic_reg dans logistic_reg_model_path. Il faudra comme pour la
    #  4ème étape passer le bon paramètre logistic_reg_model_path à la fonction dans le script main.py
    """
    On utilise la même fonction joblib que ce qui a été fait dans le preprocessing  
    """
