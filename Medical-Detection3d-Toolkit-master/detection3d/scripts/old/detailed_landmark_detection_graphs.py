import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def load_data(file_path):
    return pd.read_csv(file_path)

def plot_error_distribution(data, output_file):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='landmark', y='error_magnitude', data=data)
    plt.title('Error Magnitude Distribution by Landmark')
    plt.xlabel('Landmark')
    plt.ylabel('Error Magnitude (mm)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_3d_error_scatter(data, output_file):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data['x_error'], data['y_error'], data['z_error'], 
                         c=data['error_magnitude'], cmap='viridis')
    ax.set_xlabel('X Error (mm)')
    ax.set_ylabel('Y Error (mm)')
    ax.set_zlabel('Z Error (mm)')
    plt.colorbar(scatter, label='Error Magnitude (mm)')
    plt.title('3D Scatter Plot of Detection Errors')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_heatmap_by_image(data, output_file):
    pivot_data = data.pivot_table(values='error_magnitude', index='image', columns='landmark', aggfunc='mean')
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Mean Error Magnitude Heatmap by Image and Landmark')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_outliers(data, output_file):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='landmark', y='error_magnitude', data=data)
    plt.title('Error Magnitude Scatter Plot with Outliers')
    plt.xlabel('Landmark')
    plt.ylabel('Error Magnitude (mm)')
    plt.xticks(rotation=45, ha='right')
    
    # Annotate points with error_magnitude > 20mm
    for idx, row in data[data['error_magnitude'] > 20].iterrows():
        plt.annotate(f"{row['image']}", (row['landmark'], row['error_magnitude']))
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    data = load_data('/home/juval.gutknecht/Projects/Data/results/inference_results_aaa/report/landmark_detection_detailed.csv')
    
    plot_error_distribution(data, 'error_distribution_by_landmark.png')
    plot_3d_error_scatter(data, '3d_error_scatter.png')
    plot_heatmap_by_image(data, 'error_heatmap_by_image.png')
    plot_outliers(data, 'error_outliers.png')

if __name__ == '__main__':
    main()