import pandas as pd
import numpy  as np

from breanna import database_access    as da
from breanna.util import to_ospath

import pickle
import os
import random

from sklearn.model_selection import PredefinedSplit, cross_val_score

def create_training_set(ctr_aggregators, conn,
                        ontology, name, year, month, min_impressions,
                        viz_extractor, viz_aggregator,
                        matching_rule, verbose=True):
    
    # get aggregated CTR
    if not set(ctr_aggregators).issubset({'publisher', 'event_hour', 'os', 'device'}):
        raise ValueError('ctr_aggregators can only contain the following: \'publisher\', \'event_hour\', \'os\', \'device\'')
    ids = da.get_monthly_campaign_ids(ontology, name, year, month, conn)
    if verbose: print('Collecting click-through rate from the database...')
    ctr_agg_df = da.aggregate_ctr(ids, ctr_aggregators, min_impressions, conn)
    print(f'Finished. {len(ctr_agg_df)} examples found.')
    
    # get aggregated visual features
    banner_roots = da.get_banner_roots(ontology, name, year, month, conn)
    if verbose: print('Extracting visual features from banners...')
    viz_agg_df = da.aggregate_viz(banner_roots, viz_aggregator)
    print(f'Finished. {len(viz_agg_df)} banners preprocessed.')
    
    # join the two above dataframes
    if verbose: print('Matching banners...')
    joined = join_ctr_and_viz(ctr_agg_df, viz_agg_df, matching_rule)
    print(f'Finished. {len(joined)} examples matched.')
    
    X = joined.drop(['banner_name', 'ctr'], axis=1)
    y = joined['ctr']
    banner_names = joined['banner_name']
    return pd.get_dummies(X), y, banner_names

def join_ctr_and_viz(ctr_agg_df, viz_agg_df, matching_rule):
    # column order of the joined_df: viz_features then aggregators then ctr
    joined_df = pd.DataFrame( columns=viz_agg_df.columns.append(ctr_agg_df.columns[1:]) )
    for ctr_agg_row in ctr_agg_df.iterrows():
        for viz_agg_row in viz_agg_df.iterrows():
            ctr_row = ctr_agg_row[1]
            viz_row = viz_agg_row[1]
            if matching_rule(ctr_row['ad_name'], viz_row['banner_name']):
                joined_row = viz_row.append(ctr_row[1:])
                joined_df = joined_df.append(joined_row, ignore_index=True)
                break
    return joined_df

class CTRModel:
    
    def __init__(self, model, metadata):
        # scikit-learn model
        self.model    = model       
        self.metadata = metadata
        
    def predict(self, image):
        viz_extractor = self.metadata['viz_extractor'] 
        columns = ['feature'+str(i) for i in range(viz_extractor.get_dim())]
        for agg, vals in self.metadata['ctr_aggregators']:
            columns += [agg+'_'+val for val in vals]
        X_onehot = pd.DataFrame(columns=columns)
        n_rows = 1
        for agg, vals in self.metadata['ctr_aggregators']:
            n_rows *= len(vals)
        viz_features = viz_extractor.extract(image)
        for i in range(viz_extractor.get_dim()):
            X_onehot['feature'+str(i)] = [ viz_features[i] ] * n_rows
            
        # populate the onehot-encoded columns
        icol = viz_extractor.get_dim()
        n_consecutive_ones = n_rows
        n_repeats = 1
        for agg, vals in self.metadata['ctr_aggregators']:
            n_consecutive_ones = n_consecutive_ones // len(vals)
            period = n_consecutive_ones * len(vals)
            for j in range(len(vals)):
                col_filled = np.zeros(n_rows)
                for k in range(n_repeats):
                    col_filled[k*period+j*n_consecutive_ones : k*period+(j+1)*n_consecutive_ones] = 1
                X_onehot[columns[icol]] = col_filled
                icol += 1
            n_repeats *= len(vals)
        y_pred = self.model.predict(X_onehot)
        
        # reverse the onehot-encoding
        ctr_aggregators = [agg for agg, _ in self.metadata['ctr_aggregators']]
        X = from_dummies(X_onehot, categories=ctr_aggregators)
        X['ctr'] = y_pred
        return X[X.columns[viz_extractor.get_dim():]].copy()

# https://github.com/pandas-dev/pandas/issues/8745
# by kevin-winter
def from_dummies(data, categories, prefix_sep='_'):
    out = data.copy()
    for l in categories:
        cols, labs = [[c.replace(x,"") for c in data.columns if l+prefix_sep in c] for x in ["", l+prefix_sep]]
        out[l] = pd.Categorical(np.array(labs)[np.argmax(data[cols].values, axis=1)])
        out.drop(cols, axis=1, inplace=True)
    return out

def train_CTRModel(X_onehot, y, sklearn_estimator, 
                   viz_extractor, ctr_aggregators, verbose=True):
    metadata = create_metadata(viz_extractor, ctr_aggregators, X_onehot)
    ctrmodel = create_CTRModel(metadata, X_onehot, y, sklearn_estimator)
    return ctrmodel

def get_splits(banner_names, n_splits):
    banner_counts = banner_names.value_counts()
    random.shuffle(banner_counts)
    test_fold = np.zeros(len(banner_names))
    for i in range(n_splits):
        banners_split_i = banner_counts.index[i::n_splits]
        split_i = banner_names.isin(banners_split_i)
        test_fold[split_i] = i
    return PredefinedSplit(test_fold)

def evaluate_estimator(X_onehot, y, banner_names, sklearn_estimator, n_splits=2, n_repeats=5):
    cvscore = []
    for i in range(n_repeats):
        splits = get_splits(banner_names, n_splits)
        cvscore.append(cross_val_score(sklearn_estimator, X_onehot, y, cv=splits))
    print('R^2 = {0:.3f}(+-{1:.4f})'.format(np.mean(cvscore), np.std(cvscore)))

def create_metadata(viz_extractor, ctr_aggregators, X_onehot):
    metadata = {}
    metadata['viz_extractor']   = viz_extractor
    metadata['ctr_aggregators'] = []
    for agg in ctr_aggregators:
        vals = [col[len(agg)+1:] for col in X_onehot.columns
                if col.startswith(agg)]
        metadata['ctr_aggregators'].append((agg, vals))
    return metadata

def create_CTRModel(metadata, X_onehot, y, estimator):
    print('Training sklearn estimator...')
    model = estimator.fit(X_onehot, y)
    print('Finished.')
    ctrmodel = CTRModel(model, metadata)
    return ctrmodel

MODEL_ROOT = os.path.join(os.path.dirname(__file__),
                          to_ospath('../breanna_data/models/ctr_model'))
def save_CTRModel(ctrmodel, modelname, modelroot=MODEL_ROOT):
    with open(os.path.join(modelroot, modelname), 'wb') as output:
        pickle.dump(ctrmodel, output, pickle.HIGHEST_PROTOCOL)
        
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'CTRModel': return CTRModel
        return super().find_class(module, name)
def load_CTRModel(modelname, modelroot=MODEL_ROOT):
    with open(os.path.join(modelroot, modelname), 'rb') as input_:
        return CustomUnpickler(input_).load()