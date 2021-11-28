import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV, cross_validate, train_test_split


#################################################################################################
######## COMPACT NESTED CV and FINAL EVALUATION FUNCTIONS USING ONLY SKLEARN BUILT-IN FUNCTIONS
#################################################################################################

def inner_CV_setup(tuple_models, tuple_grids, tuple_names, k, split_method="stratified", cv_method="grid", scoring_metric="accuracy", seed=None):
    """
    Set up the inner cross validation objects used to find best hyper for each model in the inner split,
    which will be then used in the outer splits to get score on the testing set
    - tuple_models = tuple of model objects to use, for example [KNeighborsClassifier(), LogisticRegression()]
    - tuple_grids = tuple of dictionaries of hyperparameters grids to use, for example [{'n_neighbors': [15, 30, 45]}, {'penalty': ['l2'],'C': [0.5, 1, 2]}]
    - tuple_names = tuple of strings with model names, for example ["kNN", "Logreg"]
    - k = number of splits for the inner cv
    - split_method = string with CV splits method ("stratified" or "simple"), more can be added
    - cv_method = string with cross-validation technique ("grid" or "random"), more can be added
    - scoring_metric = performance metric to be used to evaluate inner cv performance (can be any metric accepted by sklearn)
    - seed = set the seed
    """
    grid_cvs = {}   # dictionary to store all the CV objects
    # define CV object
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
    """
    Perfomer Inner and Outer Cross-validation using sklearn functions.
    - grid_cvs = dictionary of sklearn Cross-Val objects (output of the function inner_CV_setup())
    - k = number of splits for the outer cv
    - X_train = train set
    - y_train = test set
    - split_method = string with CV splits method ("stratified" or "simple"), more can be added
    - scoring_metric = performance metric to be used to evaluate inner cv performance (can be any metric accepted by skleran)
    - seed = set the seed
    """
    nestedCV_results = pd.DataFrame(index=["OuterCV_average_Score%", "OuterCV_std%"])   # dataframe to store results
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
    """
    Tune hyperparamters for best model on entire training set and return it 
    (Final model evaluation outside the functions becuase in this way you can choose any score to evaluate the final performance)
    - best_model = class instance for the best model or pipeline
    - p_grid = parameters grid for the best model
    - k = number of splits for the final cv
    - X_train = train set
    - y_train = test set
    - split_method = string with CV splits method ("stratified" or "simple"), more can be added
    - cv_method = string with cross-validation technique ("grid" or "random"), more can be added
    - scoring_metric = performance metric to be used to evaluate inner cv performance (can be any metric accepted by skleran)
    - seed = set the seed
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


###################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################################



######## NESTED CV and FINAL EVALUATION FUNCTIONS with more steps shown, less built-in sklearn functions and less possible customizations 

def my_nested_CV(models, X_train, y_train, k1, k2, param_grids):
    """
    Nested cross validation, return results for each split and average score for each model in the list 'models',
    given list of parameters for the hyper tuning process ('param_grids'), 
    the train/test sets and number of split for the outer ('k1') and inner ('k2') CV.
    """

    skf_outer = StratifiedKFold(n_splits=k1, shuffle=True, random_state=111)
    nestedCV_scores = pd.DataFrame(index=[[f"Outer split {i+1}" for i in range(k1)]], columns=[models])

    # Loop for Outer and Inner splits
    for i, indexes in enumerate(skf_outer.split(X_train, y_train)):    
        X_train_cv = X_train.iloc[indexes[0]]   
        y_train_cv = y_train.iloc[indexes[0]]
        X_test_cv = X_train.iloc[indexes[1]]   
        y_test_cv = y_train.iloc[indexes[1]]
        # inner split
        skf_inner = StratifiedKFold(n_splits=k2, shuffle=True, random_state=222)
        # loop for each model to find best hyperparams and compute score given each Outer split and the associated inner splits
        for mod, grid in zip(models, param_grids):
            grCV = GridSearchCV(estimator=mod, param_grid=grid, cv=skf_inner) 
            results_grCV = grCV.fit(X_train_cv, y_train_cv)
            # identify best estimator and compute score on Outer split test set
            best_model = results_grCV.best_estimator_   
            best_model.fit(X_train_cv, y_train_cv)
            sc = best_model.score(X_test_cv, y_test_cv)    # in this case accuracy as metric for simplicity, every other score could be used (also you could add a parameter in the function to identify scoring method)
            nestedCV_scores.loc[f"Outer split {i+1}", mod] = sc
    
    nestedCV_scores.loc["Average Score"] = nestedCV_scores.mean(axis=0)  # add average score for each model at the end
    return nestedCV_scores


def my_final_evaluation(best_model, param_grid, k, X_train, y_train, X_final_test, y_final_test):
    """
    Tune hyperparameters and compute score on final evaluation set for the best model.
    Return model object with best hyperparameters and score on final evaluation set
    """
    CVsplit = StratifiedKFold(n_splits=k, shuffle=True, random_state=678)
    grCV = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=CVsplit)
    results_grCV = grCV.fit(X_train, y_train)  # entire training set
    best_hyper_best_model = results_grCV.best_estimator_
    # final score for optimized best model
    best_hyper_best_model.fit(X_train, y_train)
    score_final = best_hyper_best_model.score(X_final_test, y_final_test) 
    return best_hyper_best_model, score_final


######## TEST Functions
# list_models = [KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier()]
# dictparam1, dictparam2, dictparam3 = {"n_neighbors": [5, 20, 50]}, {"C": [0.1, 1, 2]}, {"criterion": ["gini", "entropy"], "max_depth": [20, 50]}
# list_grids = [dictparam1, dictparam2, dictparam3]

# df_results = my_nested_CV(list_models, X_train, y_train, 5, 3, list_grids)

# # assuming the best model is....
# best_model = DecisionTreeClassifier()
# best_grid = {"criterion": ["gini", "entropy"], "max_depth": [25, 50, 75, 100]}
# best_mod_tuned, final_eval_score = my_final_evaluation(best_model, best_grid, 8, X_train, y_train, X_test, y_test)

# print("\n")
# print(df_results)
# print("\n")
# print(f"Best model: {best_mod_tuned}, Final Score = {round(final_eval_score, 4)}")
# print("\n")



######## EXAMPLE FUNCTION TO PRINT THE STEPS OF THE PROCESS (simplified)
def nested_CV_example():
    """
    Show the steps of nested CV process (hyperparameters tuning and model selection):
    2 models: kNN and LogReg, 
    test 3 hyperparameters for each model (kNN: n_neighbors = 5, 20, 50; Logreg: C = 0.1, 1, 2)
    call the function to print and follow all the steps of the process
    """
    from sklearn.model_selection import train_test_split
    df = pd.read_csv('data/titanic.csv')
    Xtita = df[['Pclass', 'SibSp', 'Parch', 'Fare']]
    ytita = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(Xtita, ytita, random_state=555, shuffle=True)
    print("\n")
    print("---------------------------------------------------")
    print("---- NESTED CROSS VALIDATION PROCESS")
    print("---------------------------------------------------")
    print("===> Phase 1: OUTER & INNER CROSS VALIDATION SPLITS")
    # first split --> 5 blocks (n_splits=5)
    skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=111)
    # for each training set of the first split perform 3 sub-splits (inner) and test hyperparam (repeat for all 5 outer splits)
    score_knn = []
    score_lr = []
    for i, indexes in enumerate(skf_outer.split(X_train, y_train)):    # use enumerate to get also the split number
        # define train and test set for the 5 splits (skf_outer)
        indtr = indexes[0]    # first element of indexes (indexes is a list) = train index for the given split (i)
        indtest = indexes[1]    # second element = test index
        X_train_cv = Xtita.iloc[indtr]   
        y_train_cv = ytita.iloc[indtr]
        X_test_cv = Xtita.iloc[indtest]   
        y_test_cv = ytita.iloc[indtest]
        print(f"* OUTER split number {i+1}, using starting X and y training sets")

        # new split (3 folds)
        skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=222)
        # hyperparams grids
        p_grid_knn = {"n_neighbors": [5, 20, 50]} 
        p_grid_lr = {"C": [0.1, 1, 2]} 

        knn = KNeighborsClassifier()
        lr = LogisticRegression()
        # perform girdsearchCV using X_train_cv, y_train_cv as training set --> split in 3 new folds (skf_inner)
        GRknn = GridSearchCV(estimator=knn, param_grid=p_grid_knn, cv=skf_inner)    # could also use randomized
        GRlr = GridSearchCV(estimator=lr, param_grid=p_grid_lr, cv=skf_inner)

        # results of hyper tuning:
        # A) select best combination with .best_estimator_
        results_knn = GRknn.fit(X_train_cv, y_train_cv)
        results_lr = GRlr.fit(X_train_cv, y_train_cv)
        print(f"A) Find best hyperparameters given INNER split obtained from OUTER split {i+1}")
        print(f"Best hyper for OUTER split {i+1} given INNER-splits: kNN = {results_knn.best_estimator_}, Lr = {results_lr.best_estimator_}")
        
        # B) find accuracy of the best_estimator_ on the test set of the OUTER split number k
        # repeat for each train and test set in the outer split (1 to k)
        # knn
        print(f"B) Compute score of the best combination of hyperparameters on the TEST set of the OUTER split {i+1}")
        best_knn_k = results_knn.best_estimator_
        best_knn_k.fit(X_train_cv, y_train_cv)
        scknn = best_knn_k.score(X_test_cv, y_test_cv)
        # lr
        best_lr_k = results_lr.best_estimator_
        best_lr_k.fit(X_train_cv, y_train_cv)
        sclr = best_lr_k.score(X_test_cv, y_test_cv)
        score_knn.append(scknn)
        score_lr.append(sclr)
        print(f"Best kNN score for split {i+1}: {scknn}")
        print(f"Best Lr score for split {i+1}: {sclr}")
        print(f"End of the process for OUTER CV number {i+1}")
        print("===================================================")

    avgknn = np.mean(score_knn)
    avglr = np.mean(score_lr)
    best = ["kNN" if (avgknn-avglr)>0 else "LogReg"]
    print("\n")
    print("===> End of the NESTED CV process")
    print("***** Final results *****")
    print(f"Avg score kNN: {avgknn}")
    print(f"Avg score Lr: {avglr}")
    print(f"The best model is {best}")
    print("\n")

    print("===> Phase 2: HYPERPARAMETERS TUNING FOR THE BEST MODEL and FINAL TEST")
    if best == "kNN": 
        model = KNeighborsClassifier()
        p_grid = {"n_neighbors": [5, 20, 50, 100, 150]}   # define pram grid
    else:
        model = LogisticRegression()
        p_grid = {"C": [0.1, 1, 2, 5]} 
    print("A) Find optimal hyperparameters for the best model (use entire training set)")
    # A) find best hyper on the entire training set
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=222)
    grCV = GridSearchCV(estimator=model, param_grid=p_grid, cv=split)
    results_grCV = grCV.fit(X_train, y_train)  # entire training set
    best_hyper_best_model = results_grCV.best_estimator_
    print(f"** Best hyperparameters combination: {best_hyper_best_model}")
    # B) compute score on final evaluation set for the best model
    best_hyper_best_model.fit(X_train, y_train)
    score_final = best_hyper_best_model.score(X_test, y_test)   # use final evaluation set
    print("B) Compute score for the best model on the final evaluation set")
    print(f"** Score on the final evaluation set for the best model with tuned hyperparmaeters = {score_final}")
    print("\n")

# nested_CV_example()
