

import numpy as np
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, 
                             precision_score, recall_score, classification_report, 
                             ConfusionMatrixDisplay)
import tensorly as tl
from tensorly.decomposition import tucker
import matplotlib.pyplot as plt
import time
import sys

sys.path.append('/content/BrainTumor_clf_TDA/src')
from neural_network_trainer import train_neural_network


# UTILITY FUNCTIONS

def import_data(dataset_path, labels, image_size):
    """Import and preprocess images"""
    x_data = []
    y_data = []

    for label in labels:
        label_path = os.path.join(dataset_path, label)
        
        if not os.path.exists(label_path):
            continue

        for file in tqdm(os.listdir(label_path), desc=f"Loading {label}"):
            try:
                image = cv2.imread(os.path.join(label_path, file), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                    
                # Preprocessing pipeline
                image = cv2.GaussianBlur(image, (5, 5), 0)
                image = cv2.convertScaleAbs(image, alpha=1.5, beta=80)
                image = cv2.resize(image, (image_size, image_size))

                x_data.append(image)
                y_data.append(labels.index(label))
            except:
                continue

    return x_data, y_data


def data_to_negative(data):
    """Convert images to negative"""
    return 255 - np.array(data)


def plot_confusion_matrix(conf_matrix, classes):
    """Plot confusion matrix"""
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=classes)
    disp.plot(colorbar=False, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig('/content/BrainTumor_clf_TDA/classification_confusion_matrix.png', dpi=150)
    plt.show()


def compute_tucker_decomposition(rank, x_data):
    """Compute Tucker decomposition"""
    start_time = time.time()
    core, factors = tucker(x_data, rank)
    elapsed_time = time.time() - start_time
    print(f"Tucker decomposition time: {elapsed_time:.2f}s")
    return core, factors[0], factors[1], factors[2]


# MAIN CLASSIFICATION TRAINING

def main():
    """Train classification model"""
    
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    path_train = '/content/BrainTumor_clf_TDA/data/Dataset/Training'
    path_test = '/content/BrainTumor_clf_TDA/data/Dataset/Testing'
    
    if not os.path.exists(path_train) or not os.path.exists(path_test):
        print("ERROR: Dataset paths not found")
        return
    
    print("Loading data...")
    x_train, y_train = import_data(path_train, labels, 250)
    x_test, y_test = import_data(path_test, labels, 250)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    x_train = data_to_negative(x_train)
    x_test = data_to_negative(x_test)
    x_train = np.array(x_train) / 255.0
    x_test = np.array(x_test) / 255.0
    
    print("\nTucker decomposition...")
    X_train = np.vstack([img.flatten() for img in x_train])
    X_test = np.vstack([img.flatten() for img in x_test])
    
    x_train_fold = tl.fold(X_train, mode=2, shape=(250, 250, x_train.shape[0]))
    x_test_fold = tl.fold(X_test, mode=2, shape=(250, 250, x_test.shape[0]))
    
    rank_decomposition = (10, 10, 300)
    core, U0, U1, U2 = compute_tucker_decomposition(rank_decomposition, x_train_fold)
    
    pU2 = core @ U2.T
    core_p = tl.unfold(pU2, mode=2)
    
    step1 = tl.tenalg.mode_dot(x_test_fold, np.linalg.pinv(U0), mode=0)
    step2 = tl.tenalg.mode_dot(step1, np.linalg.pinv(U1), mode=1)
    x_test_unfolded = tl.unfold(step2, mode=2)
    
    print(f"Features: {X_train.shape} -> {core_p.shape}")
    
    print("\nPreparing data...")
    core_p_numpy = np.array(core_p, dtype=np.float32)
    x_test_numpy = np.array(x_test_unfolded, dtype=np.float32)
    y_train_numpy = np.array(y_train, dtype=np.int32)
    y_test_numpy = np.array(y_test, dtype=np.int32)
    
    print("\nTraining neural network...")
    nn_model, nn_history, nn_loss, nn_accuracy, nn_f1 = train_neural_network(
        core_p_numpy, y_train_numpy,
        x_test_numpy, y_test_numpy,
        labels=labels
    )
    
    print("\nEvaluating and saving...")
    # Get predictions
    y_pred = np.argmax(nn_model.predict(x_test_numpy, verbose=0), axis=1)
    
    cm = confusion_matrix(y_test_numpy, y_pred)
    plot_confusion_matrix(cm, labels)
        
    # Save model and preprocessing components
    import pickle
    
    model_data = {
        'model': nn_model,
        'U0': U0,
        'U1': U1,
        'U2': U2,
        'rank': rank_decomposition,
        'labels': labels,
        'accuracy': nn_accuracy,
        'f1_score': nn_f1
    }
    
    with open('/content/BrainTumor_clf_TDA/classification_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Saved: classification_model.pkl")
    print("Saved: balanced_neural_network_model.h5")
    
    # Final Summary
    
    print(f"\nFinal Performance (Summary):")
    print(f"  Accuracy:  {nn_accuracy:.4f} ({nn_accuracy*100:.2f}%)")
    print(f"  F1-Score:  {nn_f1:.4f}")
    print(f"  Loss:      {nn_loss:.4f}")
    
    print(f"\nPer-class Performance (Recall from final evaluation):")
    for i, label in enumerate(labels):
        mask = y_test_numpy == i
        if mask.sum() > 0:
            acc = accuracy_score(y_test_numpy[mask], y_pred[mask])
            print(f"  {label:12s}: {acc:.4f} ({acc*100:.2f}%)")
    
    return nn_model, nn_accuracy, nn_f1


if __name__ == "__main__":
    main()