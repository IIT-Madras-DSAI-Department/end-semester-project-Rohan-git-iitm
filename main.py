import pandas as pd
import numpy as np
import time

train = pd.read_csv('MNIST_train.csv')
val = pd.read_csv('MNIST_validation.csv') # Change this to MNIST_test.csv to test final performance.

y_train = train['label']
y_val = val['label']
x_train = train.drop(columns=['label','even'])
x_val = val.drop(columns=['label','even'])

n_classes = 10

from algorithms import XGBoostClassifier, FastWeightedKNN, PCA

def to_np(X):
    """Convert pandas DataFrame or numpy array to float32 numpy array."""
    return X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else X.astype(np.float32)

X_train = to_np(x_train)
X_val = to_np(x_val)

y_train_np = np.asarray(y_train, dtype=np.int64)
y_val_np = np.asarray(y_val, dtype=np.int64)

start = time.time()

knn_configs = [] # list of (k, n_components, weight) tuples we are using for KNN ensembles

k_list = [4, 5, 6]       # K values I am using for WeightedKNN
ncomp_list = [40, 48, 55]    # PCA components I am trying for WeightedKNN

for k in k_list:
    for n_comp in ncomp_list:
        if k == 6 and n_comp == 48:
            weight = 5.0     # boosted config since this configuration gives highest val_accuracy.
        else:
            weight = 1.0
        knn_configs.append((k, n_comp, weight))

pca_knn_list = []
knn_list = []
knn_weights = []

# Fitting each KNN model on PCA(X_train)
for k_knn, n_comp, weight in knn_configs:
    #print(f"Training PCA+KNN: k={k_knn}, n_comp={n_comp}, weight={weight}")
    # PCA on training data
    pca_knn = PCA(n_components=n_comp)
    X_train_pca = pca_knn.fit_transform(X_train)

    # Fitting KNNs:
    knn = FastWeightedKNN(k=k_knn, n_classes=n_classes)
    knn.fit(X_train_pca, y_train_np)

    pca_knn_list.append(pca_knn)
    knn_list.append(knn)
    knn_weights.append(float(weight))

knn_pred_matrix = []

for (k_knn, n_comp, weight), pca_knn, knn in zip(knn_configs, pca_knn_list, knn_list):
    X_val_pca = pca_knn.transform(X_val)
    preds_val = knn.predict(X_val_pca)
    knn_pred_matrix.append(preds_val)

knn_pred_matrix = np.vstack(knn_pred_matrix).T
N_val, n_knn_models = knn_pred_matrix.shape
knn_weights = np.asarray(knn_weights, dtype=np.float64)

# Weighted majority voting across KNN models:
knn_ensemble_pred = np.zeros(N_val, dtype=np.int64)
for i in range(N_val):
    votes = knn_pred_matrix[i]
    class_scores = np.zeros(n_classes, dtype=np.float64)
    for m in range(n_knn_models):
        label = votes[m]
        class_scores[label] += knn_weights[m]
    knn_ensemble_pred[i] = np.argmax(class_scores)

# Fitting XGBoost model per digit (one-vs-all)
xgb_models = []
#print("\nFitting 10 OvR XGBoost models...")
for digit in range(n_classes):
    # Binary target: 1 if label is this digit, else 0
    y_binary = (y_train_np == digit).astype(int)

    model = XGBoostClassifier(
        n_estimators=50,
        lamda=3,
        learning_rate=0.3,
        subsample_features=0.10
    )
    model.fit(X_train, y_binary)
    #print(f"trained XGB for class {digit}")
    xgb_models.append(model)

# Getting class scores from XGB on X_val
xgb_score_list = []
for model in xgb_models:
    score = model.predict(X_val)
    xgb_score_list.append(score)

xgb_score_val = np.vstack(xgb_score_list).T
xgb_pred_val = np.argmax(xgb_score_val, axis=1).astype(np.int64)


# Final Ensemble: KNN + XGBoost weighted voting

N_val, n_knn_models = knn_pred_matrix.shape
knn_weights = np.asarray(knn_weights, dtype=np.float64)

# Weight for XGBoost vote in the ensemble
xgb_weight = 5.0

knn_best_label = np.zeros(N_val, dtype=np.int64)
final_pred_vote = np.zeros(N_val, dtype=np.int64)

for i in range(N_val):
    votes = knn_pred_matrix[i]
    class_scores = np.zeros(n_classes)

    # KNN weighted voting
    for m in range(n_knn_models):
        class_scores[votes[m]] += knn_weights[m]
    knn_best_label[i] = np.argmax(class_scores)

    # Adding XGBoost vote to KNN votes:
    xgb_label = xgb_pred_val[i]
    class_scores[xgb_label] += xgb_weight

    # Final ensemble prediction (KNN + XGB vote)
    final_pred_vote[i] = np.argmax(class_scores)

# Accuracies
knn_ensemble_acc = np.mean(knn_best_label == y_val_np)
xgb_full_acc = np.mean(xgb_pred_val == y_val_np)
final_acc_vote = np.mean(final_pred_vote == y_val_np)

end = time.time()

# All the statistics printed:
print(f"\nTime taken                      : {end - start:.2f} seconds")
print(f"WeightedKNN ensemble accuracy   : {knn_ensemble_acc * 100:.2f}%")
print(f"XGB One vs Rest accuracy        : {xgb_full_acc * 100:.2f}%")
print(f"Final ensemble accuracy         : {final_acc_vote * 100:.2f}% (xgb_weight={xgb_weight})")


