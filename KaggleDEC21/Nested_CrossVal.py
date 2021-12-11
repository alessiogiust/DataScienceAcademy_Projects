import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV, cross_validate


def inner_CV_setup(tuple_models, tuple_grids, tuple_names, k, split_method="stratified", cv_method="grid", scoring_metric="accuracy", seed=None):
    """ Set up the inner cross-validation objects used to find best hyperparameters for each model in the inner split,
    which will be then used in the outer splits to get score on the testing set.

    Parameters
    ----------
    tuple_models: tuple of model objects 
        ML models or Pipelines to to be used, for example (KNeighborsClassifier(), LogisticRegression()) or (pipe1, pipe2)
    tuple_grids: tuple of dictionaries 
        Dicts of hyperparameter grids to use in the hyperparam tuning process, for example ([{'n_neighbors': [15, 30, 45]}], [{'penalty': ['l2'],'C': [0.5, 1, 2]}])
    tuple_names: tuple of strings 
        Names of the model, for example ("kNN", "Logreg")
    k: int
        Number of splits for the inner cv
    split_method: string, default = "stratified"
        Possible CV splits methods (alternative is "simple"), more can be added
    cv_method: string, default = "grid"
        Possible cross-validation techniques (alternative is "random"), more can be added
    scoring_metric: string, default = "accuracy"
        Performance metric to evaluate inner cv performance (can be any metric accepted by sklearn)
    seed: int
        Set the seed for random CV splits 
    
    Returns
    -------
    grid_cvs: dictionary
        Dictionary with (key) = tuple_names element, (value) = GridSearchCV/RandomizedSearchCV object for each model 
    """
    grid_cvs = {}   
    if split_method == "stratified": 
        inner_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed) 
    else: 
        inner_cv = KFold(n_splits=k, shuffle=True, random_state=seed) 
    for mod, grid, name in zip(tuple_models, tuple_grids, tuple_names):
        if cv_method == "grid": 
            gcv = GridSearchCV(estimator=mod, param_grid=grid, scoring=scoring_metric, n_jobs=-1, cv=inner_cv, refit=True, error_score="raise")   # set refit to True to use the best model in the inner split to get score on the outer test set
        else: 
            gcv = RandomizedSearchCV(estimator=mod, param_grid=grid, scoring=scoring_metric, n_jobs=-1, cv=inner_cv, refit=True, error_score="raise")
        grid_cvs[name] = gcv
    return grid_cvs


def nestedCV_model_selection(grid_cvs, k, X_train, y_train, split_method="stratified", scoring_metric="accuracy", seed=None):
    """ Perfomer Inner and Outer Cross-validation using sklearn functions.

    Parameters
    ----------
    grid_cvs: dictionary 
        Dictionary produced by inner_CV_setup(), with sklearn Cross-Val objects
    k: int
        Number of splits for the outer cv
    X_train, y_train: pandas dataframes with X and y for the models
    split_method: string, default = "stratified"
        Possible CV splits methods (alternative is "simple"), more can be added
    scoring_metric: string, default = "accuracy"
        Performance metric to evaluate inner cv performance (can be any metric accepted by sklearn)
    seed: int

    Returns
    -------
    nestedCV_results: pandas dataframe
        Average score and standard deviation of the score obtain from the Nested CV process for each model (model names as columns).
    """
    nestedCV_results = pd.DataFrame(index=["OuterCV_average_Score%", "OuterCV_std%"])  
    if split_method == "stratified": 
        outer_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    else: 
        outer_cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    for name, gs_est in sorted(grid_cvs.items()):   # loop trough the dictionary produced by the inner_CV_setup() function
        results = cross_validate(gs_est, X=X_train, y=y_train, scoring=scoring_metric, cv=outer_cv, n_jobs=-1)
        scores = results["test_score"]
        nestedCV_results.loc["OuterCV_average_Score%", name] = scores.mean() * 100
        nestedCV_results.loc["OuterCV_std%", name] = scores.std() * 100 
    return nestedCV_results


def final_tuning(best_model, p_grid, k, X_train, y_train, split_method="stratified", cv_method="grid", scoring_metric="accuracy", seed=None):
    """ Tune hyperparameters for best model on entire training set
    (Final model evaluation outside the functions because in this way you can choose any score to evaluate the final performance)
    
    Parameters
    ----------
    best_model: sklearn model object or Pipeline object 
        Class instance for the best model or Pipeline
    p_grid: dictionary
        Parameters grid for the best model
    k: int
        Number of splits for the final cv
    X_train, y_train: pandas dataframes with X and y for the model
    split_method: string, default = "stratified"
        Possible CV splits methods (alternative is "simple"), more can be added
    cv_method: string, default = "grid"
        Possible cross-validation techniques (alternative is "random"), more can be added
    scoring_metric: string, default = "accuracy"
        Performance metric to evaluate inner cv performance (can be any metric accepted by sklearn)
    seed: int

    Returns
    -------
    final_best: sklearn model object or Pipeline object 
        Best model (or Pipeline for the model) tuned
    """
    if split_method == "stratified": 
        final_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed) 
    else: 
        final_cv = KFold(n_splits=k, shuffle=True, random_state=seed) 
    if cv_method == "grid": 
        gcv_model_select = GridSearchCV(estimator=best_model, param_grid=p_grid, scoring=scoring_metric, n_jobs=-1, cv=final_cv, refit=True)
    else: 
        gcv_model_select = RandomizedSearchCV(estimator=best_model, param_grid=p_grid, scoring=scoring_metric, n_jobs=-1, cv=final_cv, refit=True)
    gcv_model_select.fit(X_train, y_train)
    final_best = gcv_model_select.best_estimator_
    return final_best


######## FULL NESTED CV INITIALIZATION + PROCESS (w/ and w/out pipeline) ########
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# ## 0) Importing data, defining training (and final evaluation test (OPTIONAL))
# df = pd.read_csv('dataML/titanic.csv')
# Xtita = df[['Pclass', 'SibSp', 'Parch', 'Fare']]
# ytita = df['Survived']
# X_train, X_test, y_train, y_test = train_test_split(Xtita, ytita, random_state=555, shuffle=True)
# ### 1) Initializing Classifiers and Pipelines
# knn = Pipeline([("scaler", MinMaxScaler()), ("model1", KNeighborsClassifier())])
# lr = Pipeline([("scaler", MinMaxScaler()), ("model2", LogisticRegression())])
# tree = DecisionTreeClassifier(random_state=1)
# all_models = (knn, lr, tree)
# names_models = ("kNN", "LogReg", "Tree")
# ### 3) Setting up the parameters grids  (small to be fast)
# param_grid1 = [{'model1__n_neighbors': [15, 30, 45]}]
# param_grid2 = [{'model2__penalty': ['l2'],'model2__C': [0.5, 1, 2]}]
# param_grid3 = [{'max_depth': [15, 50, None], 'criterion': ['gini', 'entropy']}]
# all_grids = (param_grid1, param_grid2, param_grid3)
# ### 4) Setting up the inner CVs
# cvs_dict = inner_CV_setup(all_models, all_grids, names_models, 2, scoring_metric="precision")
# ### 5) Nested Cross-Validation
# nestedresults_df = nestedCV_model_selection(cvs_dict, 3, X_train, y_train, scoring_metric="precision")
# print("\n")
# print(nestedresults_df)
# ### 6) Select best model and Tune its Hyperparameters
# best_mod = Pipeline([("scaler", MinMaxScaler()), ("model1", KNeighborsClassifier())])  # for example...
# best_mod_tuned = final_tuning(best_mod, param_grid1, 3, X_train, y_train, scoring_metric="precision")
# print("\n")
# print(f"Best model after final tuning: {best_mod_tuned}")
# print("\n")
# ### 7) (OPTIONAL) Compute performance on final evaluation for any score
# best_mod_tuned.fit(X_train, y_train)
# final_sc = best_mod_tuned.score(X_test, y_test)
# print(f"Score for {best_mod_tuned} on final evaluation set = {final_sc}")
# print("\n")
# ### 8) Final model to be used in production (train on all the data available)
# prod_model = best_mod_tuned.fit(Xtita, ytita)
