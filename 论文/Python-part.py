def objective(hyperparameters):
    '''Returns validation score from hyperparameters'''
     model = Classifier(hyperparameters)
    validation_loss = cross_validation(model, training_data)
    return validation_loss


import lightgbm as lgb
from hyperopt import STATUS_OK
N_FOLDS = 10
# Create the dataset
train_set = lgb.Dataset(train_features, train_labels)
def objective(params, n_folds=N_FOLDS):
   '''Objective function for Gradient Boosting Machine Hyperparameter Tuning'''
   # Perform n_fold cross validation with hyperparameters
   # Use early stopping and evalute based on ROC AUC
   cv_results = lgb.cv(params, train_set, nfold=n_folds, num_boost_round=10000,
                       early_stopping_rounds=100, metrics='auc', seed=50)
   # Extract the best score
   best_score = max(cv_results['auc-mean'])
   # Loss must be minimized
   loss = 1 - best_score
   # Dictionary with information for evaluation
   return {'loss': loss, 'params': params, 'status': STATUS_OK}


import lgb
# Default gradient boosting machine classifier
model = lgb.LGBMClassifier()
model
LGBMClassifier(boosting_type='gbdt', n_estimators=100,
              class_weight=None, colsample_bytree=1.0,
              learning_rate=0.1, max_depth=-1,
              min_child_samples=20,
              min_child_weight=0.001, min_split_gain=0.0,
              n_jobs=-1, num_leaves=31, objective=None,
              random_state=None, reg_alpha=0.0, reg_lambda=0.0,
              silent=True, subsample=1.0,
              subsample_for_bin=200000, subsample_freq=1)

from hyperopt import hp
# Discrete uniform distribution
num_leaves = {'num_leaves': hp.quniform('num_leaves', 30, 150, 1)}


# Learning rate log uniform distribution
learning_rate = {'learning_rate': hp.loguniform('learning_rate',
                                                np.log(0.005),
                                                 np.log(0.2)}


# Define the search space
space = {
   'class_weight': hp.choice('class_weight', [None, 'balanced']),
   'boosting_type': hp.choice('boosting_type',
                              [{'boosting_type': 'gbdt',
                                   'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                {'boosting_type': 'dart',
                                    'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                {'boosting_type': 'goss'}]),
'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
   'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
   'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
   'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
   'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
   'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
   'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}


# choice: category variable
# quniform: discrete uniform (integer interval uniform)
# uniform: continuous and uniform (interval is a floating point number)
# loguniform: continuous logarithmic uniform (evenly distributed under logarithm)
# boosting type domain
boosting_type = {'boosting_type': hp.choice('boosting_type',
                                           [{'boosting_type': 'gbdt',
                                                 'subsample': hp.uniform('subsample', 0.5, 1)},
                                            {'boosting_type': 'dart',
                                                 'subsample': hp.uniform('subsample', 0.5, 1)},
                                            {'boosting_type': 'goss',
                                                 'subsample': 1.0}])}
# Sample from the full space
example = sample(space)

# Dictionary get method with default
subsample = example['boosting_type'].get('subsample', 1.0)

# Assign top-level keys
example['boosting_type'] = example['boosting_type']['boosting_type']
example['subsample'] = subsample

example
{'boosting_type': 'gbdt',
'class_weight': 'balanced',
'colsample_bytree': 0.8111305579351727,
'learning_rate': 0.16186471096789776,
'min_child_samples': 470.0,
'num_leaves': 88.0,
'reg_alpha': 0.6338327001528129,
'reg_lambda': 0.8554826167886239,
'subsample_for_bin': 280000.0,
'subsample': 0.6318665053932255}

from hyperopt import tpe
# Algorithm
tpe_algorithm = tpe.suggest
from hyperopt import Trials
# Trials object to track progress
bayes_trials = Trials()
# In order to monitor the training progress,
# the result history can be written into the CSV file
# to prevent the evaluation result from disappearing due to accidental interruption of the program.
import csv
# File to save first results
out_file = 'gbm_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()
# Then, in the objective function, we can add rows to write to CSV at each iteration:
1 # Write to the csv file ('a' means append)
2 of_connection = open(out_file, 'a')
3 writer = csv.writer(of_connection)
4 writer.writerow([loss, params, iteration, n_estimators, run_time])
5 of_connection.close()


from hyperopt import fmin
MAX_EVALS = 500
# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, 
            max_evals = MAX_EVALS, trials = bayes_trials)


{'boosting_type': 'gbdt',
  'class_weight': 'balanced',
  'colsample_bytree': 0.7125187075392453,
  'learning_rate': 0.022592570862044956,
  'min_child_samples': 250,
  'num_leaves': 49,
  'reg_alpha': 0.2035211643104735,
  'reg_lambda': 0.6455131715928091,
  'subsample': 0.983566228071919,
  'subsample_for_bin': 200000}


The best model scores 0.72506 AUC ROC on the test set.
The best cross validation score was 0.77101 AUC ROC.
This was achieved after 413 search iterations.
