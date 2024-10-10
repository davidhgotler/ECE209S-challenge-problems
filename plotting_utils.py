import numpy as np
from matplotlib import pyplot as plt
from gridworld import label_str

def plot_pmatrix(p_matrix, title, dims:tuple[int]=(5,5)):
    grid_matrix = p_matrix.reshape(dims)

    state_labels = np.array(list(label_str[:dims[0]*dims[1]])).reshape(dims)

    # Plot the heatmap
    plt.imshow(grid_matrix, cmap='viridis', interpolation='nearest',origin='lower')
    plt.yticks(np.arange(4,-1,-1))

    # Add annotations for each cell with both the state label and probability value
    for i in range(dims[0]):
        for j in range(dims[1]):
            # Display state label and probability in each cell
            plt.text(j, i, f"{state_labels[i, j]}\n{grid_matrix[i, j]:.2f}",
                    ha='center', va='center', color='white' if grid_matrix[i, j] < 0.5 else 'black')

    # Add title and format axes
    plt.title(title)
    plt.xticks(ticks=np.arange(dims[1]+1)-0.5, labels=np.arange(dims[1]+1))  # Show x-axis ticks
    plt.yticks(ticks=np.arange(dims[1]+1)-0.5, labels=np.arange(dims[1]+1))

    plt.show()