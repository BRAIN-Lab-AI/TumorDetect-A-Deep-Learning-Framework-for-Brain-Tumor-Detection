import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def create_gradcam_cnn(input_shape=(250, 250, 1), num_classes=4):
    """CNN architecture optimized for Grad-CAM visualization"""
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Last convolutional block for Grad-CAM
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax', name='predictions')
    ], name='GradCAM_CNN')
    
    return model


def plot_training_history(history):
    """Plot training and validation loss/accuracy"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss During Training', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy During Training', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/content/TumorDetect-A-Deep-Learning-Framework-for-Brain-Tumor-Detection/cnn_training_history.png', dpi=150)
    plt.show()
    
    print("Saved: cnn_training_history.png")


def plot_confusion_matrix(conf_matrix, classes):
    """Plot confusion matrix"""
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=classes)
    disp.plot(colorbar=False, cmap='Blues')
    plt.title("Confusion Matrix - CNN Classification")
    plt.savefig('/content/TumorDetect-A-Deep-Learning-Framework-for-Brain-Tumor-Detection/cnn_confusion_matrix.png', dpi=150)
    plt.show()
    
    print("Saved: cnn_confusion_matrix.png")


def load_data(dataset_path, labels, image_size=250):
    """Load and preprocess brain tumor images"""
    
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
                image = 255 - image
                image = image / 255.0

                x_data.append(image)
                y_data.append(labels.index(label))
            except:
                continue

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_data = np.expand_dims(x_data, axis=-1)
    
    return x_data, y_data


class GradCAM:
    """Grad-CAM implementation for CNN visualization"""

    def __init__(self, model, layer_name='conv4_2'):
        self.model = model
        self.layer_name = layer_name

        if not hasattr(model, "inputs") or model.inputs is None:
            dummy_input = tf.zeros((1, 250, 250, 1), dtype=tf.float32)
            _ = model(dummy_input)

        conv_layer = model.get_layer(layer_name)
        self.grad_model = keras.Model(
            inputs=model.input,
            outputs=[conv_layer.output, model.output]
        )

    def compute_heatmap(self, image, class_idx, eps=1e-8):
        """Generate Grad-CAM heatmap for given class"""
        
        if not isinstance(image, tf.Tensor):
            image = tf.convert_to_tensor(image, dtype=tf.float32)
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image, training=False)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + eps)
        
        return heatmap.numpy()

    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """Overlay heatmap on original image"""
        
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(
                (image * 255).astype(np.uint8),
                cv2.COLOR_GRAY2RGB
            )
        else:
            image_rgb = (image * 255).astype(np.uint8)
        
        overlay = cv2.addWeighted(image_rgb, 1-alpha, heatmap_colored, alpha, 0)
        return heatmap_resized, overlay


def visualize_gradcam_results(gradcam, images, predictions, true_labels, 
                              class_names, num_samples=12):
    """Create Grad-CAM visualization grid"""
    
    num_samples = min(num_samples, len(images))
    rows = (num_samples + 2) // 3
    
    fig, axes = plt.subplots(rows, 12, figsize=(24, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    print("Generating Grad-CAM heatmaps...")
    
    for idx in tqdm(range(num_samples)):
        row = idx // 3
        col_offset = (idx % 3) * 4
        
        pred_idx = predictions[idx]
        true_idx = true_labels[idx]
        
        img_original = images[idx, :, :, 0]
        img_batch = np.expand_dims(images[idx], axis=0)
        
        heatmap = gradcam.compute_heatmap(img_batch, pred_idx)
        heatmap_resized, overlay = gradcam.overlay_heatmap(img_original, heatmap)
        
        axes[row, col_offset].imshow(img_original, cmap='gray')
        axes[row, col_offset].set_title(f'Original\n{class_names[true_idx]}', fontsize=9)
        axes[row, col_offset].axis('off')
        
        axes[row, col_offset+1].imshow(heatmap_resized, cmap='jet')
        axes[row, col_offset+1].set_title('Grad-CAM', fontsize=9)
        axes[row, col_offset+1].axis('off')
        
        axes[row, col_offset+2].imshow(overlay)
        axes[row, col_offset+2].set_title('Localization', fontsize=9)
        axes[row, col_offset+2].axis('off')
        
        status = 'Correct' if pred_idx == true_idx else 'Wrong'
        color = 'green' if pred_idx == true_idx else 'red'
        axes[row, col_offset+3].text(0.5, 0.5, f'{status}\n{class_names[pred_idx]}',
                                     ha='center', va='center',
                                     fontsize=11, color=color, fontweight='bold')
        axes[row, col_offset+3].axis('off')
    
    plt.suptitle('Grad-CAM Tumor Detection', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/content/TumorDetect-A-Deep-Learning-Framework-for-Brain-Tumor-Detection/gradcam_detection_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Saved: gradcam_detection_results.png")


def main():
    """Train CNN and generate visualizations"""
    
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    path_train = '/content/TumorDetect-A-Deep-Learning-Framework-for-Brain-Tumor-Detection/data/Dataset/Training'
    path_test = '/content/TumorDetect-A-Deep-Learning-Framework-for-Brain-Tumor-Detection/data/Dataset/Testing'
    
    if not os.path.exists(path_train) or not os.path.exists(path_test):
        print("ERROR: Dataset paths not found")
        return
    
    print("Loading data...")
    x_train, y_train = load_data(path_train, labels)
    x_test, y_test = load_data(path_test, labels)
    
    print(f"Training: {x_train.shape}")
    print(f"Testing: {x_test.shape}")
    
    print("Building model...")
    model = create_gradcam_cnn(input_shape=(250, 250, 1), num_classes=len(labels))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )
    
    print("Training...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    print("Plotting training history...")
    plot_training_history(history)
    
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    
    print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=labels, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    plot_confusion_matrix(cm, labels)
    
    print("\nPer-class Performance:")
    for i, label in enumerate(labels):
        mask = y_test == i
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], predictions[mask])
            print(f"  {label:12s}: {acc:.4f} ({acc*100:.2f}%)")
    
    model.save('/content/TumorDetect-A-Deep-Learning-Framework-for-Brain-Tumor-Detection/cnn_for_gradcam.h5')
    print("\nModel saved: cnn_for_gradcam.h5")
    
    print(f"\nFinal Performance:")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Loss: {test_loss:.4f}")
    
    return model, history


if __name__ == "__main__":
    main()
