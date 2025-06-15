import matplotlib.pyplot as plt
import numpy as np

def plot_explained_variance(explained_variance, title='Explained Variance'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.title(title)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.grid()
    plt.show()

def plot_principal_components(components, labels=None, title='Principal Components'):
    plt.figure(figsize=(10, 6))
    plt.scatter(components[:, 0], components[:, 1], alpha=0.7)
    
    if labels is not None:
        for i, label in enumerate(labels):
            plt.annotate(label, (components[i, 0], components[i, 1]), fontsize=8)

    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.show()