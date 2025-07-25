import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_metrics(plot_df, metric: str, config_order: list, title: str, x_label: str, y_label: str, legend_loc: str, output_folder: str):
    """
    Args:
        plot_df (pd.DataFrame): pandas DataFrame with columns ['Seed', 'Editing Config', 'IoU', 'Dice', 'ASD', 'HD']. 
        metric (str): the metric to plot (e.g., 'Dice', 'ASD', 'HD').
        config_order (list): the order of editing configurations for the x-axis.
        title (str): plot title.
        x_label (str): label for the x-axis.
        y_label (str): label for the y-axis.
        legend_loc (str): location of the legend in the plot.
        output_folder (str): Folder where the plot will be saved.
    """

    # Reorder the 'Editing Config' 
    plot_df['Editing Config'] = pd.Categorical(plot_df['Editing Config'], categories=config_order, ordered=True)
    #print(plot_df['Editing Config'].dtypes) #should be category
    #print(plot_df['Editing Config'].unique())

    # Compute statistics
    stats = plot_df.groupby('Editing Config')[metric].agg(['min', 'max', 'mean']).reset_index()

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # 1. Plot per-seed lines
    sns.lineplot(
        data=plot_df,
        x='Editing Config',
        y=metric,
        hue='Seed',
        palette='pastel',
        linewidth=1,
        alpha=0.7,
        #marker='o', #markersize=5,
        legend=False
    )

    # 2. Plot Mean line
    sns.lineplot(
        data=stats,
        x='Editing Config',
        y='mean',
        color='#D73027',
        linewidth=2,
        marker='o',
        markersize=6,
        label='Mean'
    )

    # 3. Min-max shaded range
    plt.fill_between(
        stats['Editing Config'],
        stats['min'],
        stats['max'],
        color='#F08080',
        alpha=0.2,
        label='Min-Max Range'
    )

    # Formatting
    plt.title(title, fontsize=16, weight='bold', pad=20)
    plt.xlabel(x_label, fontsize=13)
    plt.xticks(fontsize=11)
    plt.ylabel(y_label, fontsize=13)
    plt.yticks(fontsize=11)
    plt.ylim(stats['min'].min() - 0.01, stats['max'].max() + 0.01)
    plt.legend(loc=legend_loc, fontsize=11, frameon=True)
    plt.tight_layout()

    #plt.show()
    # Save the plot
    os.makedirs(output_folder, exist_ok=True)
    filename = f"{title.replace(' ', '_')}.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {filepath}")




def plot_metrics_with_config_extraction(df: pd.DataFrame,
                                        metric: str,
                                        config_order: list,
                                        title: str,
                                        x_label: str,
                                        y_label: str,
                                        legend_loc: str,
                                        output_folder: str):
    """
    Plots a metric across editing configurations extracted from `edited_subfolder`.

    Args:
        df (pd.DataFrame): Must contain ['seed', 'edited_subfolder', metric].
        metric (str): Metric to plot (e.g., 'Dice', 'IoU').
        config_order (list): Order of configurations (e.g., ['0.8_0.6', ...]).
        title (str): Plot title.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        legend_loc (str): Legend location (e.g., 'lower right').
        output_folder (str): Folder to save the plot.
    """

    # ✅ 1. Extract simplified config
    def extract_config(name: str) -> str:
        parts = name.split("_")
        return f"{int(parts[1]) / 10}_{int(parts[2]) / 10}"

    df = df.copy()
    df.rename(columns={"patient_id": "Seed"}, inplace=True)
    df["Editing Config"] = df["edited_subfolder"].apply(extract_config)

    # ✅ 2. Apply categorical ordering
    df["Editing Config"] = pd.Categorical(
        df["Editing Config"],
        categories=config_order,
        ordered=True
    )

    # Compute stats
    stats = df.groupby("Editing Config")[metric].agg(['min', 'max', 'mean']).reset_index()

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Per-seed lines
    sns.lineplot(
        data=df,
        x="Editing Config",
        y=metric,
        hue="Seed",
        palette="pastel",
        linewidth=1,
        alpha=0.7,
        legend=False
    )

    # Mean line
    sns.lineplot(
        data=stats,
        x="Editing Config",
        y="mean",
        color="#D73027",
        linewidth=2,
        marker="o",
        markersize=6,
        label="Mean"
    )

    # Min-max shaded range
    plt.fill_between(
        stats["Editing Config"],
        stats["min"],
        stats["max"],
        color="#F08080",
        alpha=0.2,
        label="Min-Max Range"
    )

    # Formatting
    plt.title(title, fontsize=16, weight="bold", pad=20)
    plt.xlabel(x_label, fontsize=13)
    plt.ylabel(y_label, fontsize=13)
    plt.xticks(fontsize=11, rotation=15)
    plt.yticks(fontsize=11)
    plt.ylim(stats["min"].min() - 0.01, stats["max"].max() + 0.01)
    plt.legend(loc=legend_loc, fontsize=11, frameon=True)
    plt.tight_layout()

    #Save plot
    os.makedirs(output_folder, exist_ok=True)
    filename = f"{title.replace(' ', '_')}.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {filepath}")