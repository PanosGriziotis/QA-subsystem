import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_retrievers_test_report():
    json_path = "reports/retrievers_eval_report_npho20.json"
    
    # Check if the JSON file exists
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path) as fp:
        data = json.load(fp)

    # Extract data for plotting
    methods = ['bm25', 'embedding_retriever', 'dpr']
    metrics = ['recall', 'map', 'mrr', 'retrieve_time']
    plot_data = {metric: {method: [] for method in methods} for metric in metrics}
    top_k_values = list(range(1, 21))  # top_k from 1 to 20

    for method in methods:
        for k in top_k_values:
            entry = data[method][str(k)]
            for metric in metrics:
                plot_data[metric][method].append(entry[metric])

    # Create the plots
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Performance Metrics for Retrieval Methods', fontsize=16)

    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        for method in methods:
            ax.plot(top_k_values, plot_data[metric][method], label=method)
        ax.set_title(metric.capitalize())
        ax.set_xlabel('Top K')
        ax.set_ylabel(metric.capitalize())
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot as an image file
    output_path = 'performance_metrics.png'
    plt.savefig(output_path)
    print(f"Plot saved as {output_path}")


if __name__ == "__main__":
    plot_retrievers_test_report()