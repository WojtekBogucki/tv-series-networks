from networks.utils import *
import os
import matplotlib
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# data
episode_stats = pd.DataFrame()
for show_name in ["the_office", "seinfeld", "tbbt", "friends"]:
    net_episodes = get_episode_networks(f"../data/{show_name}/")
    latest_file = [f for f in os.listdir(f"../data/{show_name}/") if f.startswith(f"{show_name}_lines_v")][-1]
    episode_dict = get_episode_dict(f"../data/{show_name}/{latest_file}")
    ep_stats = get_network_stats_by_episode(net_episodes, episode_dict, show_name)
    ep_stats["show"] = show_name
    episode_stats = pd.concat([episode_stats, ep_stats], axis=0)

episode_stats = episode_stats.fillna(0)
# episode_stats = episode_stats[episode_stats["runtime"]<30]

# data preparing
X = episode_stats.drop(['avg_rating', 'num_votes', 'runtime', 'viewership', 'show'], axis=1).to_numpy()
y = episode_stats["avg_rating"].to_numpy()
feature_names = episode_stats.drop(['avg_rating', 'num_votes', 'runtime', 'viewership', 'show'], axis=1).columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model
rf_base = RandomForestRegressor()

# grid search
n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
max_features = [None, 'sqrt']
max_depth = [None, 5, 10, 20, 50, 100, 150]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


rf_random = RandomizedSearchCV(estimator=rf_base, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)

rf_random.fit(X_train, y_train)

rf_random.best_params_
rf_random.best_score_
rf_best = rf_random.best_estimator_

rf_best.score(X_train, y_train)
rf_best.score(X_test, y_test)
y_pred = rf_best.predict(X_test)
r2_score(y_test, y_pred)
mean_absolute_error(y_test, y_pred)
mean_squared_error(y_test, y_pred)

rf_base.fit(X_train, y_train)

rf_base.score(X_train, y_train)
rf_base.score(X_test, y_test)
y_pred_base = rf_base.predict(X_test)
r2_score(y_test, y_pred_base)
mean_absolute_error(y_test, y_pred_base)
mean_squared_error(y_test, y_pred_base)

# MDI feature importance
importances = rf_best.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_best.estimators_], axis=0)
mdi_importances = pd.Series(importances, index=feature_names)
impo_sort = mdi_importances.argsort()
mdi_importances = mdi_importances[impo_sort]
std = std[impo_sort]

fig, ax = plt.subplots(figsize=(20, 10))
mdi_importances.plot.barh(xerr=std, ax=ax)
ax.set_title("Feature importances using MDI", fontsize=20)
ax.set_xlabel("Mean decrease in impurity", fontsize=14)
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
fig.tight_layout()
plt.savefig(f"../figures/comparison/rf_feat_imp_impurity")

# permutation feature importance
result = permutation_importance(rf_best, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_importances = pd.Series(result.importances_mean, index=feature_names)
perm_sort = perm_importances.argsort()
perm_importances = perm_importances[perm_sort]
perm_std = result.importances_std[perm_sort]

fig, ax = plt.subplots(figsize=(20, 10))
perm_importances.plot.barh(xerr=perm_std, ax=ax)
ax.set_title("Feature importances using permutation on full model", fontsize=20)
ax.set_xlabel("Mean decrease of coefficient of determination", fontsize=14)
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
fig.tight_layout()
plt.savefig(f"../figures/comparison/rf_feat_imp_perm")

