import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

print("Loading model...")
model = keras.models.load_model('/content/BrainTumor_clf_TDA/cnn_for_gradcam.h5')
print("Model loaded")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    """Generate Grad-CAM heatmap using gradient tape"""
    
    conv_layer = model.get_layer(last_conv_layer_name)
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        
        x = img_tensor
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_output = x
        
        final_output = x
        class_score = final_output[:, pred_index]
    
    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_output = conv_output[0]
    pooled_grads = pooled_grads.numpy()
    conv_output = conv_output.numpy()
    
    heatmap = np.zeros(conv_output.shape[:2], dtype=np.float32)
    for i in range(len(pooled_grads)):
        heatmap += pooled_grads[i] * conv_output[:, :, i]
    
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    
    return heatmap


def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(np.uint8(255 * img), cv2.COLOR_GRAY2RGB)
    else:
        img = np.uint8(255 * img)
    
    superimposed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    
    return heatmap, superimposed


print("Loading test images...")
path_test = '/content/BrainTumor_clf_TDA/data/Dataset/Testing'

x_test = []
y_test = []

for label_idx, label in enumerate(labels):
    label_path = os.path.join(path_test, label)
    files = sorted(os.listdir(label_path))[:3]
    
    for file in files:
        img = cv2.imread(os.path.join(label_path, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=80)
        img = cv2.resize(img, (250, 250))
        img = 255 - img
        img = img / 255.0
        
        x_test.append(img)
        y_test.append(label_idx)

x_test = np.array(x_test)
y_test = np.array(y_test)
x_test_cnn = np.expand_dims(x_test, axis=-1)

print(f"Loaded {len(x_test)} test images")

print("Getting predictions...")
predictions = np.argmax(model.predict(x_test_cnn, verbose=0), axis=1)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("Generating Grad-CAM heatmaps...")

fig, axes = plt.subplots(4, 12, figsize=(24, 16))

for idx in tqdm(range(12)):
    row = idx // 3
    col_offset = (idx % 3) * 4
    
    pred_idx = predictions[idx]
    true_idx = y_test[idx]
    
    img_original = x_test[idx]
    img_batch = np.expand_dims(x_test_cnn[idx], axis=0)
    
    heatmap = make_gradcam_heatmap(img_batch, model, 'conv4_2', pred_idx)
    heatmap_colored, overlay = overlay_heatmap_on_image(img_original, heatmap)
    
    axes[row, col_offset].imshow(img_original, cmap='gray')
    axes[row, col_offset].set_title(f'Original\n{labels[true_idx]}', fontsize=9)
    axes[row, col_offset].axis('off')
    
    axes[row, col_offset+1].imshow(heatmap, cmap='jet')
    axes[row, col_offset+1].set_title('Grad-CAM', fontsize=9)
    axes[row, col_offset+1].axis('off')
    
    axes[row, col_offset+2].imshow(overlay)
    axes[row, col_offset+2].set_title('Localization', fontsize=9)
    axes[row, col_offset+2].axis('off')
    
    status = 'Correct' if pred_idx == true_idx else 'Wrong'
    color = 'green' if pred_idx == true_idx else 'red'
    axes[row, col_offset+3].text(0.5, 0.5, f'{status}\n{labels[pred_idx]}',
                                 ha='center', va='center',
                                 fontsize=11, color=color, fontweight='bold')
    axes[row, col_offset+3].axis('off')

plt.suptitle(f'Grad-CAM Tumor Detection ({accuracy*100:.2f}% Accuracy)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/content/BrainTumor_clf_TDA/gradcam_detection_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nSaved: gradcam_detection_results.png")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nPer-class Performance:")
for i, label in enumerate(labels):
    mask = y_test == i
    if mask.sum() > 0:
        acc = (predictions[mask] == y_test[mask]).mean()
        print(f"  {label:12s}: {acc:.4f} ({acc*100:.2f}%)")
