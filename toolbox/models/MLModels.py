from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


"""
    Build an SVM model using scikit-learn.

    Parameters:
    - X_train: The feature matrix of the training data.
    - y_train: The labels of the training data.
    - kernel: The kernel type to be used in the algorithm (default: 'linear').
    - C: Regularization parameter (default: 1.0).
    - gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' (default: 'scale').
    - random_state: Seed for random number generation (default: None).

    Returns:
    - svm_model: The trained SVM model.
    """
def SVM(kernel):
    model = svm.SVC()
    return model

def RF():
    model = RandomForestClassifier()
    return model