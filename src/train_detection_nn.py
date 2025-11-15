import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorly as tl
import pickle


class InputGradientDetection:
    """Compute gradients from prediction back to input image"""
    
    def __init__(self, model, U0, U1, U2):
        self.model = model
        self.U0 = tf.constant(U0, dtype=tf.float32)
        self.U1 = tf.constant(U1, dtype=tf.float32)
        self.U2 = tf.constant(U2, dtype=tf.float32)
        print("Input Gradient Detection initialized")
    
    def apply_tucker_transform(self, image):
        """Apply Tucker transformation"""
        
        image_flat = tf.reshape(image, [-1])
        image_tensor = tf.reshape(image_flat, [250, 250, 1])
        
        U0_p = tf.linalg.pinv(self.U0)
        U1_p = tf.linalg.pinv(self.U1)

        step1 = tf.einsum('ri,ijk->rjk', U0_p, image_tensor)
        step2 = tf.einsum('sj,rjk->rsk', U1_p, step1)
        features = tf.reshape(step2, [-1])
        
        return features
    
    def compute_saliency_map(self, image, class_idx):
        """Compute pixel importance via gradients"""
        
        image = tf.constant(image, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(image)
            
            features = self.apply_tucker_transform(image)
            features = tf.expand_dims(features, 0)
            
            predictions = self.model(features, training=False)
            target_class = predictions[0, class_idx]
        
        grads = tape.gradient(target_class, image)
        
        if grads is None:
            return np.zeros((250, 250))
        
        saliency = tf.abs(grads).numpy()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def overlay_saliency(self, original_image, saliency, alpha=0.4):
        """Overlay saliency map on image"""
        
        saliency_colored = cv2.applyColorMap(
            (saliency * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        if len(original_image.shape) == 2:
            original_rgb = cv2.cvtColor(
                (original_image * 255).astype(np.uint8),
                cv2.COLOR_GRAY2RGB
            )
        else:
            original_rgb = (original_image * 255).astype(np.uint8)
        
        overlay = cv2.addWeighted(original_rgb, 1-alpha, saliency_colored, alpha, 0)
        
        return saliency, overlay


def visualize_detection(detector, images, predictions, true_labels, class_names, num_samples=12):
    """Visualize detection results"""
    
    num_samples = min(num_samples, len(images))
    rows = (num_samples + 2) // 3
    
    fig, axes = plt.subplots(rows, 12, figsize=(24, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    print("Generating saliency maps...")
    
    for idx in tqdm(range(num_samples)):
        row = idx // 3
        col_offset = (idx % 3) * 4
        
        pred_idx = predictions[idx]
        true_idx = true_labels[idx]
        
        saliency = detector.compute_saliency_map(images[idx], pred_idx)
        _, overlay = detector.overlay_saliency(images[idx], saliency)
        
        axes[row, col_offset].imshow(images[idx], cmap='gray')
        axes[row, col_offset].set_title(f'Original\n{class_names[true_idx]}', fontsize=8)
        axes[row, col_offset].axis('off')
        
        axes[row, col_offset+1].imshow(saliency, cmap='jet')
        axes[row, col_offset+1].set_title('Detection', fontsize=8)
        axes[row, col_offset+1].axis('off')
        
        axes[row, col_offset+2].imshow(overlay)
        axes[row, col_offset+2].set_title('Localization', fontsize=8)
        axes[row, col_offset+2].axis('off')
        
        status = 'Correct' if pred_idx == true_idx else 'Wrong'
        color = 'green' if pred_idx == true_idx else 'red'
        axes[row, col_offset+3].text(0.5, 0.5, f'{status}\n{class_names[pred_idx]}',
                                     ha='center', va='center',
                                     fontsize=12, color=color, fontweight='bold')
        axes[row, col_offset+3].axis('off')
    
    plt.suptitle('Tumor Detection - Input Gradients', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    for ax in axes.flat:
        if not ax.has_data():
            ax.axis('off')

    plt.savefig('/content/BrainTumor_clf_TDA/detection_nn_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Saved: detection_nn_results.png")


def import_data(dataset_path, labels, image_size):
    """Load images"""
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
                    
                image = cv2.GaussianBlur(image, (5, 5), 0)
                image = cv2.convertScaleAbs(image, alpha=1.5, beta=80)
                image = cv2.resize(image, (image_size, image_size))

                x_data.append(image)
                y_data.append(labels.index(label))
            except:
                continue

    return np.array(x_data), np.array(y_data)


def data_to_negative(data):
    """Convert to negative"""
    return 255 - data


def main():
    """Generate detection visualization"""
    
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    path_test = '/content/BrainTumor_clf_TDA/data/Dataset/Testing'
    model_path = '/content/BrainTumor_clf_TDA/classification_model.pkl'
    nn_model_path = '/content/BrainTumor_clf_TDA/balanced_neural_network_model.h5'
    
    if not os.path.exists(model_path) or not os.path.exists(nn_model_path):
        print("ERROR: Model files not found")
        print("Run train_classification.py first")
        return
    
    print("Loading model...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    U0 = model_data['U0']
    U1 = model_data['U1']
    U2 = model_data['U2']
    
    nn_model = keras.models.load_model(nn_model_path)
    
    print(f"Model loaded (accuracy: {model_data['accuracy']:.4f})")
    
    print("\nLoading test images...")
    x_test, y_test = import_data(path_test, labels, 250)
    
    x_test = data_to_negative(x_test)
    x_test = x_test / 255.0
    
    print(f"Loaded {len(x_test)} images")
    
    print("\nGetting predictions...")
    X_test = np.vstack([img.flatten() for img in x_test])
    x_test_fold = tl.fold(X_test, mode=2, shape=(250, 250, x_test.shape[0]))
    
    step1 = tl.tenalg.mode_dot(x_test_fold, np.linalg.pinv(U0), mode=0)
    step2 = tl.tenalg.mode_dot(step1, np.linalg.pinv(U1), mode=1)
    x_test_features = tl.unfold(step2, mode=2)
    x_test_features = np.array(x_test_features, dtype=np.float32)
    
    predictions = np.argmax(nn_model.predict(x_test_features, verbose=0), axis=1)
    accuracy = np.mean(predictions == y_test)
    
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nGenerating detection visualization...")
    detector = InputGradientDetection(nn_model, U0, U1, U2)
    
    from sklearn.utils import shuffle
    x_test, y_test, predictions = shuffle(x_test, y_test, predictions, random_state=42)
    
    samples_per_class = 5
    indices = []
    for label_idx in range(len(labels)):
        idxs = np.where(y_test == label_idx)[0]
        take = min(samples_per_class, len(idxs))
        chosen = np.random.choice(idxs, take, replace=False)
        indices.extend(chosen)

    indices = np.random.permutation(indices)
    
    visualize_detection(detector, x_test[indices], predictions[indices], 
                       y_test[indices], labels, num_samples=len(indices))
    
    print("\nPer-class Performance:")
    for i, name in enumerate(labels):
        idxs = np.where(y_test == i)[0]
        acc = (predictions[idxs] == y_test[idxs]).mean()
        correct = (predictions[idxs] == y_test[idxs]).sum()
        total = len(idxs)
        print(f"  {name:12s}: {acc:.4f} ({correct}/{total})")
    
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    return detector


if __name__ == "__main__":
    main()