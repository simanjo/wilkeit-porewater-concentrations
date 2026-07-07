import time
import os
import pickle
from urllib.error import HTTPError
from urllib.request import urlopen
from urllib.parse import quote
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_validate
from sklearn.pipeline import Pipeline


KRR_GRID = {
        'regressor__alpha': np.logspace(-5, 2, 8),
        'regressor__kernel': ['rbf', 'laplacian', 'sigmoid'],
        'regressor__gamma': np.logspace(-4, 0, 5),
    }

PLS_GRID = {
    'regressor__n_components': np.arange(2, 7),
    'regressor__max_iter': [5_000],
}

RF_GRID = {
    'regressor__n_estimators': [10, 50],
    'regressor__criterion': ['absolute_error', 'friedman_mse'],
    'regressor__min_samples_split': [2, 4],
    'regressor__min_samples_leaf': [1, 2, 4],
}

GB_GRID = {
    'regressor__n_estimators': [10, 50, 100],
    'regressor__loss': ['squared_error', 'huber'],
    'regressor__learning_rate': [0.01, 0.1, 1, 10],
    'regressor__max_depth': [5, None],
}

# Copyright rapelpy CC-BY-SA
# Modified from https://stackoverflow.com/a/54932071
@cache
def get_smiles(name: str, pubchem_only: bool=True) -> str | None:
    pubchem_url = None
    smiles = None
    if pubchem_only:
        pubchem_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + quote(name) + '/property/CanonicalSMILES/TXT'
    else:
        url = 'https://cactus.nci.nih.gov/chemical/structure/' + quote(name) + '/smiles'
        try:
            smiles = urlopen(url).read().decode('utf8').strip()
            smiles = str(smiles)
        except HTTPError as e:
            print(f"Encountered an error while parsing {name}: {e}")
            print("Trying pubchem for fallback")
            pubchem_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + quote(name) + '/property/CanonicalSMILES/TXT'
    if pubchem_url:
        try:
            smiles = urlopen(pubchem_url).read().decode('utf8').strip()
            smiles = str(smiles)
            # if there are multiple smiles only take the first
            smiles = smiles.split("\n")[0]
        except HTTPError as e:
            print(f"Encountered an HTTP error while parsing {name} in pubchem: {e}")
            if "HTTP Error 400: PUGREST.BadRequest" in str(e):
                # sleep shortly to throttle requests
                time.sleep(0.2)
                print("Retrying with smiles field")
                pubchem_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + quote(name) + '/property/SMILES/TXT'
                try:
                    smiles = urlopen(pubchem_url).read().decode('utf8').strip()
                    smiles = str(smiles)
                except HTTPError as e:
                    print(f"Could not parse {name} in pubchem: {e}")
                    smiles = None
    # sleep shortly to throttle requests
    time.sleep(0.2)
    return smiles


@cache
def get_abrahams(smiles, abrahams_data):
    abrahams_df = pd.read_csv(abrahams_data, sep=";", header=0, decimal=",")
    params = abrahams_df.loc[
        abrahams_df["Input SMILES"] == smiles
    ].iloc[0]
    return params[["E","S","A","B","V","L"]]


def get_model_path(model, target, save_dir, sobol=False):
    model_name = ""
    if isinstance(model, PLSRegression):
        model_name += "pls_"
    elif isinstance(model, KernelRidge):
        model_name += "krr_"
    elif isinstance(model, RandomForestRegressor):
        model_name += "rf_"
    elif isinstance(model, GradientBoostingRegressor):
        model_name += "gb_"
    else:
        raise NotImplementedError(
            "Please specify a valid model class (one of PLSRegression, " \
            "KernelRidge, RandomForestRegressor or GradientBoostingRegressor" \
            f"), was {model}."
        )

    model_name += target.strip().replace(" ", "_").replace("10^", "e").replace("/", "per")
    if sobol:
        model_name += "_sobol_indices"
    model_name += ".pkl"
    return Path(save_dir) / model_name
    

def train_looc(
        model, grid, target, save_dir, X, y,
        save=True, cv=LeaveOneOut()):
    if save:
        save_path = get_model_path(model, target, save_dir)
        if save_path.is_file():
            with open(save_path, 'rb') as fh:
                score, clf = pickle.load(fh) 
            return score
        os.makedirs(save_path.parent, exist_ok=True)

    start = time.time()
    pipe = Pipeline([
        ('scaling', StandardScaler()),
        ('regressor', model)
    ])
    clf = GridSearchCV(
        estimator=pipe, param_grid=grid,
        cv=cv, n_jobs=-1, error_score='raise',
        scoring='neg_root_mean_squared_error',
        return_train_score=True, refit=True
    )
    score = cross_validate(
        clf, X=X, y=y, cv=cv,
        scoring='neg_root_mean_squared_error',
        return_train_score=True,
        return_estimator=True,
        return_indices=True # type: ignore
    )
    print(f"Finished {model} for {target} in {time.time()-start} seconds.")
    if save:
        with open(save_path, 'wb') as fh:
            pickle.dump((score, clf), fh)
    return score


def split_scores(score_dict):
    scores = pd.Series(score_dict).reset_index() 
    scores.columns = ['model', 'target', 'score_dict']
    scores['test_score'] = scores['score_dict'].apply(lambda x: x['test_score'])
    return scores.drop(columns=['score_dict'])