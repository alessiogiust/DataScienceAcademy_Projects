import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV,  cross_val_score, cross_validate, train_test_split
from sklearn.utils import shuffle

df = pd.read_csv('data/titanic.csv')
Xtita = df[['Pclass', 'SibSp', 'Parch', 'Fare']]
ytita = df['Survived']

# Define training and final evaluation test
X_train, X_test, y_train, y_test = train_test_split(Xtita, ytita, random_state=555, shuffle=True)

##### NESTED CV and FINAL EVALUATION FUNCTIONS
def nested_CV(models, X_train, y_train, k1, k2, param_grids):
    # models, p_grids, X_train, y_train, k1, k2
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
            sc = best_model.score(X_test_cv, y_test_cv)    # in this case accuracy for simplicity, every other score could be used (also you could add a parameter in the function to identify scoring method)
            nestedCV_scores.loc[f"Outer split {i+1}", mod] = sc
    
    nestedCV_scores.loc["Average Score"] = nestedCV_scores.mean(axis=0)  # add average score for each model at the end
    return nestedCV_scores


def final_evaluation(best_model, param_grid, k, X_train, y_train, X_final_test, y_final_test):
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
    score_final = best_hyper_best_model.score(X_test, y_test) 
    return best_hyper_best_model, score_final


######## TEST Functions
list_models = [KNeighborsClassifier(), LogisticRegression(), DecisionTreeClassifier()]
dictparam1, dictparam2, dictparam3 = {"n_neighbors": [5, 20, 50]}, {"C": [0.1, 1, 2]}, {"criterion": ["gini", "entropy"], "max_depth": [20, 50]}
list_grids = [dictparam1, dictparam2, dictparam3]

df_results = nested_CV(list_models, X_train, y_train, 5, 3, list_grids)

# assuming the best model is....
best_model = DecisionTreeClassifier()
best_grid = {"criterion": ["gini", "entropy"], "max_depth": [25, 50, 75, 100]}
best_mod_tuned, final_eval_score = final_evaluation(best_model, best_grid, 8, X_train, y_train, X_test, y_test)

print("\n")
print(df_results)
print("\n")
print(f"Best model: {best_mod_tuned}, Final Score = {round(final_eval_score, 4)}")
print("\n")


################################################################################
######## EXAMPLE FUNCTION TO PRINT THE STEPS OF THE PROCESS (simplified)
################################################################################
def nested_CV_example():
    """
    Show the steps of nested CV process (hyperparameters tuning and model selection):
    2 models: kNN and LogReg, 
    test 3 hyperparameters for each model (kNN: n_neighbors = 5, 20, 50; Logreg: C = 0.1, 1, 2)
    call the function to print and follow all the steps of the process
    """
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
