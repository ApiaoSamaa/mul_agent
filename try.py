# %%
import numpy as np
import streamlit as st
import pandas as pd

points = 10
# change dataframe raw name, using data.index, directly change it.
df = pd.DataFrame(data=np.random.randn(points,points),columns=(i+1 for i in range(points)))
df.index = df.columns

# print(df.index)
# Int64Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='int64')

df 
# print(df.shape)
#(10,10)
# df[i] for i in range(df.shape[0])

df_np = np.array(df)
idx_diag = np.diag_indices(df_np.shape[0])
df_np[idx_diag] = 0
df = pd.DataFrame(df_np.astype('int32'))

df

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure((8*8))
fig.add_subplot(1,2,1)
plt.show()
# %%


from PIL import Image
import io
import math

def plot_fully_connected_graph(num_nodes):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Generate the coordinates for each node
    radius = 1
    node_coords = [(math.cos(i * 2 * math.pi / num_nodes) * radius, math.sin(i * 2 * math.pi / num_nodes) * radius) for i in range(num_nodes)]

    # Plot the nodes
    for i, coords in enumerate(node_coords):
        ax.plot(coords[0], coords[1], 'ro')
        ax.annotate(f'Node {i+1}', coords, textcoords="offset points", xytext=(0,10), ha='center')

    # Plot the edges (connections between nodes)
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            ax.plot([node_coords[i][0], node_coords[j][0]], [node_coords[i][1], node_coords[j][1]], 'b-')

    # Set axis limits and labels
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Save the figure to a file
    fig.savefig('graph.png')

    plt.show()
    # Convert the plot to an image
    fig.canvas.draw()
    image = Image.open(io.BytesIO(fig.canvas.tostring_rgb()))
    plt.close(fig)

    # Display the image in Streamlit
    st.image(image=image)


# Example usage with Streamlit
num_nodes = st.slider('Number of nodes', min_value=2, max_value=10, value=4)
plot_fully_connected_graph(8)

# %%
