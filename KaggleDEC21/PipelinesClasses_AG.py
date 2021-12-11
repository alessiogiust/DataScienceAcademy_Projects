import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from Nested_CrossVal import inner_CV_setup, nestedCV_model_selection
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def reduce_mem_usage(df, verbose=True):
    """ 
    Reduce memory uage for the given dataframe by changing data types (!! function from Kaggle discussions)
    """
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def preprocessDataset(dataset, pct=0.001, seed=123): 
    """ Obtain a random sub-sample from the original dataframe and drop rows where Cover_Type = 4, 5, 6

    Parameters
    ----------
    dataset: pandas dataframe
    pct: float, default = 0.001
        Percentage of entire dataset to use as random sub-sample

    Returns
    -------
    newdf: pandas dataframe 
    """
    newdf = dataset.sample(frac=pct, random_state=seed)
    newdf = newdf[newdf["Cover_Type"] != 5]
    newdf = newdf[newdf["Cover_Type"] != 4]
    newdf = newdf[newdf["Cover_Type"] != 6]
    return newdf


class FixFeatures(BaseEstimator, TransformerMixin):
    """ Fix feature ranges. The class could be generalized 

    Attributes
    ----------
    fit: do nothing
    transform: fix Aspect and Hillshade features, returns fixed pandas dataframe
    """
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["Aspect"][df["Aspect"] < 0] += 360
        df["Aspect"][df["Aspect"] > 359] -= 360
        df["Aspect"][df["Aspect"] < 0] += 360
        df["Aspect"][df["Aspect"] > 359] -= 360
        df.loc[df["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
        df.loc[df["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
        df.loc[df["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
        df.loc[df["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
        df.loc[df["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
        df.loc[df["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255
        return df


class FeaturesEng(BaseEstimator, TransformerMixin):
    """ Feature engeneering. The class could be generalized 

    Attributes
    ----------
    fit: do nothing
    transform: create new features (Soil_type_count, Wilderness_area_count, mnhttn_dist_hydrlgy, Hillshade_mean, amp_Hillshade). Returns pandas dataframe
    """
    def __init__(self):
        return None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Identify soil and wilderness variables
        soil_features = [x for x in df.columns if x.startswith("Soil_Type")]
        wilderness_features = [x for x in df.columns if x.startswith("Wilderness_Area")]
        features_Hillshade = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
        df["Soil_type_count"] = df[soil_features].sum(axis=1)
        df["Wilderness_area_count"] = df[wilderness_features].sum(axis=1)
        # Manhhattan distance to Hydrology
        df["mnhttn_dist_hydrlgy"] = np.abs(df["Horizontal_Distance_To_Hydrology"]) + np.abs(df["Vertical_Distance_To_Hydrology"])
        # Avergae Hillshade
        df["Hillshade_mean"] = df[features_Hillshade].mean(axis=1)
        # Max - Min Hillshade
        df['amp_Hillshade'] = df[features_Hillshade].max(axis=1) - df[features_Hillshade].min(axis=1)
        df = df.dropna()
        return df


class UseSelectedFeatures(BaseEstimator, TransformerMixin):
    """ Use only features identified during personalized features selection process.

    Parameters
    ----------
    selected_feat: list of strings
        List of selected features

    Attributes
    ----------
    fit: do nothing
    transform: return pandas dataframe with only selected features
    """
    def __init__(self, selected_feat):
        self.selected_feat = selected_feat
        return None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df = df[self.selected_feat] 
        return df


def pipeline_NestedCV(X_train, y_train, feat_keep):
    """
    Nested Cross-Validation process (see Nested_CrossVal.py)
    Create X_train and y_train after preprocessing with reduce_mem_usage() and preprocessDataset()
        
    Parameters
    ----------
    X_train: pandas dataframe
    y_train: pandas dataframe
    feat_keep: list strings
        Features to use in the training process.

    Returns
    -------
    nestedCV_results: pandas dataframe
        Average score and standard deviation of the score obtain from the Nexted CV process for each model.
    """
    # Prepare pipelines
    # change the parameteres, models, k-fold splits, etc... as you wish
    pipe_knn = Pipeline([("fix", FixFeatures()), ("eng", FeaturesEng()), ("select", UseSelectedFeatures(feat_keep)), ("scale", MinMaxScaler()), ("est1", KNeighborsClassifier())])
    pipe_rf = Pipeline([("fix", FixFeatures()), ("eng", FeaturesEng()), ("select", UseSelectedFeatures(feat_keep)), ("est2", RandomForestClassifier(class_weight="balanced_subsample"))])
    pipe_lda = Pipeline([("fix", FixFeatures()), ("eng", FeaturesEng()), ("select", UseSelectedFeatures(feat_keep)), ("scale", MinMaxScaler()), ("est3", LinearDiscriminantAnalysis())])
    pipe_svc = Pipeline([("fix", FixFeatures()), ("eng", FeaturesEng()), ("select", UseSelectedFeatures(feat_keep)), ("scale", MinMaxScaler()), ("est4", SVC(probability=True, class_weight = "balanced"))])
    pipe_xgb = Pipeline([("fix", FixFeatures()), ("eng", FeaturesEng()), ("select", UseSelectedFeatures(feat_keep)), ("est5", XGBClassifier())])
    # pipe_knn = Pipeline([("scale", MinMaxScaler()), ("est1", KNeighborsClassifier())])
    # pipe_rf = Pipeline([("est2", RandomForestClassifier(class_weight="balanced_subsample"))])
    # pipe_lda = Pipeline([("scale", MinMaxScaler()), ("est3", LinearDiscriminantAnalysis())])
    # pipe_svc = Pipeline([("scale", MinMaxScaler()), ("est4", SVC(probability=True, class_weight = "balanced"))])
    # pipe_xgb = Pipeline([("est5", XGBClassifier())])
    models = (pipe_knn, pipe_rf, pipe_lda, pipe_svc, pipe_xgb) 

    # Define Grids
    param_grids = ([{"est1__n_neighbors": [5, 10, 25, 50], "est1__weights": ["uniform", "distance"]}],
                [{"est2__n_estimators": [100, 150, 300], "est2__criterion": ["gini", "entropy"]}],
                [{"est3__shrinkage": [None]}],
                [{"est4__C": [0.5, 1, 5], "est4__gamma": ["auto", "scale"]}],
                [{"est5__gamma": [0, 1], "est5__n_estimators": [500, 1500], "est5__subsample": [0.5, 0.85]}])

    mod_names = ("kNN", "RandomForest", "LDA", "SVC", "XGBoost")

    # Inner cv setup
    dict_CVgrids = inner_CV_setup(models, param_grids, mod_names, 3, seed=33)  # stratifiedKfold and GridSearchCV for inner CV
    # Nested CV
    nestedCV_results = nestedCV_model_selection(dict_CVgrids, 3, X_train, y_train, seed=22)  # stratifiedKfold for outer CV
    return nestedCV_results
