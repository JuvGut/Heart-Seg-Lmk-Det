
#%%

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def read_csv(file_path):
    data = pd.read_csv(file_path)
    if len(data) != 9:
        raise ValueError(f"Expected 9 points, but found {len(data)} in {file_path}")
    return data

def plot_3d_coordinates(ax, data, color, marker, label):
    scatter = ax.scatter(data['x'], data['y'], data['z'], c=color, marker=marker, label=label)
    for i, txt in enumerate(data['name']):
        ax.text(data['x'].iloc[i], data['y'].iloc[i], data['z'].iloc[i], txt)
    return scatter

def calculate_transformation(data1, data2):
    centroid1 = np.mean(data1[['x', 'y', 'z']].values, axis=0)
    centroid2 = np.mean(data2[['x', 'y', 'z']].values, axis=0)
    translation = centroid2 - centroid1
    centered1 = data1[['x', 'y', 'z']].values - centroid1
    centered2 = data2[['x', 'y', 'z']].values - centroid2
    H = centered1.T @ centered2
    U, _, Vt = np.linalg.svd(H)
    rotation = Vt.T @ U.T
    return rotation, translation

def apply_transformation(data, rotation, translation):
    transformed = (data[['x', 'y', 'z']].values @ rotation.T) + translation
    return pd.DataFrame({'name': data['name'], 'x': transformed[:, 0], 'y': transformed[:, 1], 'z': transformed[:, 2]})

def plot_connections(ax, data, color):
    lines = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            line, = ax.plot([data['x'].iloc[i], data['x'].iloc[j]], 
                            [data['y'].iloc[i], data['y'].iloc[j]], 
                            [data['z'].iloc[i], data['z'].iloc[j]], 
                            c=color, alpha=0.3)
            lines.append(line)
    return lines

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def main(file1, file2):
    data1 = read_csv(file1)
    data2 = read_csv(file2)
    rotation, translation = calculate_transformation(data1, data2)
    data1_transformed = apply_transformation(data1, rotation, translation)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter1 = plot_3d_coordinates(ax, data1, 'blue', 'o', 'Original (File 1)')
    scatter2 = plot_3d_coordinates(ax, data1_transformed, 'green', '^', 'Transformed (File 1)')
    scatter3 = plot_3d_coordinates(ax, data2, 'red', 's', 'Target (File 2)')

    plot_connections(ax, data1, 'blue')
    plot_connections(ax, data1_transformed, 'green')
    plot_connections(ax, data2, 'red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Interactive 3D Coordinate Visualization and Transformation (9 Points)')
    ax.legend()

    # Set view limits to encompass all points
    all_points = pd.concat([data1, data2, data1_transformed])
    x_min, x_max = all_points['x'].min(), all_points['x'].max()
    y_min, y_max = all_points['y'].min(), all_points['y'].max()
    z_min, z_max = all_points['z'].min(), all_points['z'].max()

    # Add padding
    padding = 0.1  # 10% padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)

    # Ensure equal aspect ratio
    set_axes_equal(ax)

    # Enable rotation
    ax.mouse_init(rotate_btn=1, zoom_btn=3)

    plt.show()

    print("Rotation Matrix:")
    np.set_printoptions(precision=5, suppress=True)
    print(rotation)
    print("\nTranslation Vector:")
    np.set_printoptions(precision=2, suppress=True)
    print(translation)

if __name__ == "__main__":
    file1 = "/home/juval.gutknecht/Projects/Data/A_Subset_012/landmarksTr_csv/BS-005_11_Herz__0.6__I26f__3__BestDiast_67_%.csv"
    file2 = "/home/juval.gutknecht/Projects/Data/A_Subset_012_a/landmarksTr_csv/BS-005_11_Herz__0.6__I26f__3__BestDiast_67_%.csv"
    main(file1, file2)