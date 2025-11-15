# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import time

def create_neural_network(input_dim, num_classes=4):
    """
    ARCHITECTURE 
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        layers.Dense(512, activation='relu', 
                    kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.45),
        
        layers.Dense(384, activation='relu',
                    kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        
        layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.0001)),
        layers.Dropout(0.15),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_neural_network(core_p, y_train, x_test_unfolded, y_test, 
                         labels=['glioma', 'meningioma', 'notumor', 'pituitary']):
    print("="*60)
    print("NEURAL NETWORK CONFIGURATION")
    print("="*60)
    
    input_dim = core_p.shape[1]
    print(f"\nInput dimensions: {input_dim}")
    print(f"Number of classes: {len(labels)}")
    print(f"Training samples: {core_p.shape[0]}")
    print(f"Test samples: {x_test_unfolded.shape[0]}")
    
    print("\n  NEURAL NETWORK Configuration:")
    print("  Architecture: [512, 384, 256, 128]")
    print("  Dropout: [0.45, 0.35, 0.25, 0.15]")
    print("  L2 Regularization: 0.0001")
    print("  Batch size: 24")
    print("  Optimizer: AdamW")
    print("  Epochs: 150 with early stopping")
    
    model = create_neural_network(input_dim, num_classes=len(labels))
    
    # AdamW optimizer
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks with patience=20
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True, verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1
    )
    
    print("\n" + "="*60)
    print("Training with Gradient Descent Optimization...")
    print("="*60)
    
    start_time = time.time()
    
    # Batch size: 24
    history = model.fit(
        core_p, y_train,
        validation_data=(x_test_unfolded, y_test),
        epochs=150,
        batch_size=24,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    test_loss, test_accuracy = model.evaluate(x_test_unfolded, y_test, verbose=0)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    y_pred_proba = model.predict(x_test_unfolded, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nWeighted F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels, digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
        
    print(f"\n Neural Network Results:")
    print(f"   Results:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    
    print("="*60)
    
    plot_training_history(history)
    
    model.save('/content/BrainTumor_clf_TDA/balanced_neural_network_model.h5')
    print("\n Model saved as '/content/BrainTumor_clf_TDA/balanced_neural_network_model.h5'")
    
    return model, history, test_loss, test_accuracy, f1

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss During Training (Shows Optimization!)', fontsize=14, fontweight='bold')
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
    plt.show()