import os 
import numpy as np 
import json
import matplotlib.pyplot as plt

# plt.rc('text', usetex=True)
plt.rc('font', family='serif')

history_path = "./history"
paths = [path for path in os.listdir(history_path)]
kvasir_paths = [path for path in paths if "_kvasir" in path]
bagls_paths = [path for path in paths if "_kvasir" not in path]

def filter_paths(paths, current_num_nests, current_num_filters, current_operation):
    filtered_paths = [path for path in paths if f"history_{current_num_nests}_{current_num_filters}_{current_operation}" in path]
    return filtered_paths

# Use the function to filter the paths
current_num_nests = 2
current_num_filters = 16
current_operation = 'add'
filtered_paths = filter_paths(paths, current_num_nests, current_num_filters, current_operation)

fig, axes = plt.subplots(2, figsize=(12, 12))

for path in bagls_paths: # only plotting BAGLS results! use either filtered_paths or paths to plot both. 
    with open(os.path.join(history_path, path), 'r') as f:
        history = json.load(f)

    # dice loss 
    axes[0].plot(history['loss'], label='Training Loss for ' + path, color='blue')
    axes[0].plot(history['val_loss'], label='Validation Loss for ' + path, color='red')

    # IoU in the second subplot
    axes[1].plot(history['iou_score'], label='Training IoU for ' + path, color='blue')
    axes[1].plot(history['val_iou_score'], label='Validation IoU for ' + path, color='red')

# title, x-axis label, and y-axis label of the dice losssubplot
axes[0].set_title('Loss for All History Files', fontsize=16)
axes[0].set_xlabel('Epochs', fontsize=14)
axes[0].set_ylabel('Loss', fontsize=14)

# title, x-axis label, and y-axis label of the IoU subplot
axes[1].set_title('IoU for All History Files', fontsize=16)
axes[1].set_xlabel('Epochs', fontsize=14)
axes[1].set_ylabel('IoU', fontsize=14)

# Add a legend to each subplot
axes[0].legend(fontsize=12)
axes[1].legend(fontsize=12)

# Add gridlines to each subplot
axes[0].grid(True)
axes[1].grid(True)

plt.tight_layout()
plt.show()