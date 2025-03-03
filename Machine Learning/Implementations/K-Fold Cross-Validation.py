import numpy as np

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True, random_seed=None):
    n_sample = len(X)
    indices = np.arange(len(X))

    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)

    fold_size = n_sample // k

    result = []
    for i in range(k):
        start = fold_size * i
        end = start + fold_size
        val_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start],indices[end:]))
        x_train,y_train = X[train_indices],y[train_indices]
        x_val,y_val = X[val_indices],y[val_indices]
        result.append((x_train.tolist(),x_val.tolist()))

    return result

print(k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=5, shuffle=False))