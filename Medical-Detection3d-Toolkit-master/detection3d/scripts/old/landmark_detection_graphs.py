import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def plot_detection_rate(data, output_file):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(data['landmark_name'], data['detection_rate (%)'])
    plt.title('Detection Rate by Landmark')
    plt.xlabel('Landmark')
    plt.ylabel('Detection Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 100)
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_error_metrics(data, output_file):
    plt.figure(figsize=(12, 6))
    x = range(len(data))
    plt.bar(x, data['mean_error'], width=0.4, label='Mean Error', align='center')
    plt.bar([i+0.4 for i in x], data['median_error'], width=0.4, label='Median Error', align='center')
    plt.title('Mean and Median Error by Landmark')
    plt.xlabel('Landmark')
    plt.ylabel('Error (mm)')
    plt.xticks([i+0.2 for i in x], data['landmark_name'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_error_distribution(data, output_file):
    plt.figure(figsize=(12, 6))
    plt.boxplot([data['percentile_25'], data['median_error'], data['percentile_75'], data['percentile_90']],
                labels=['25th Percentile', 'Median', '75th Percentile', '90th Percentile'])
    plt.title('Error Distribution Across All Landmarks')
    plt.ylabel('Error (mm)')
    plt.yscale('log')  # Using log scale due to large range of values
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_inlier_rate_vs_mean_error(data, output_file):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['inlier_rate_5mm'], data['mean_error'])
    plt.title('Inlier Rate vs Mean Error')
    plt.xlabel('Inlier Rate (5mm) (%)')
    plt.ylabel('Mean Error (mm)')
    for i, row in data.iterrows():
        plt.annotate(row['landmark_name'], (row['inlier_rate_5mm'], row['mean_error']))
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    # Load the data
    data = load_data('/home/juval.gutknecht/Projects/Data/results/inference_model_comparison/newmask_012_202020/report/landmark_detection_report.csv')

    # Generate plots
    plot_detection_rate(data, 'detection_rate_by_landmark.png')
    plot_error_metrics(data, 'error_metrics_by_landmark.png')
    plot_error_distribution(data, 'error_distribution.png')
    plot_inlier_rate_vs_mean_error(data, 'inlier_rate_vs_mean_error.png')

if __name__ == '__main__':
    main()