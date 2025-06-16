import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams.update({'font.size': 12})

font_dirs = ['/scratch4/jgray21/dxu39/miniforge3/envs/diffenergy/fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# set font
plt.rcParams['font.family'] = 'Arial'

def plot_success_rate(df):
    # Plot setup
    methods = df['method'].unique()
    metrics = df['metric'].unique()
    qualities = df['quality'].unique()
    n_metrics = len(metrics)
    n_qualities = len(qualities)
    bar_width = 0.2

    x = np.arange(len(methods))  # base x positions per method
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    # Color palette
    palette = sns.color_palette("colorblind", n_colors=n_qualities)

    for i, metric in enumerate(metrics):
        for j, quality in enumerate(qualities):
            subset = df[(df['metric'] == metric) & (df['quality'] == quality)]
            # Calculate x positions with dodging
            offset = (i - 1) * bar_width + (j - 0.5) * (bar_width / n_qualities)
            xpos = x + offset
            heights = [subset[(subset['method'] == m)]['success_rate'].values[0] for m in methods]
            axes.bar(xpos, heights,
                    width=bar_width / n_qualities,
                    label=f'{metric} | {quality}',
                    color=palette[j],
                    edgecolor='black')

    # Label setup
    axes.set_xticks(x)
    axes.set_xticklabels(methods)
    axes.set_ylabel("Success Rate")
    axes.set_xlabel("Method")
    axes.legend(title="Metric | Quality", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'success_rate.png', dpi=300, bbox_inches='tight')

def plot_success(df):
    df["method_metric"] = df["method"] + "\n" + df["metric"]

    # Define custom colors for pastel red, muted red, and dark red
    custom_palette = ["#FFC3A0", "#FF5733", "#8B0000"]

    g = sns.catplot(
        data=df, 
        kind='bar', 
        x='method_metric', 
        y='success_rate', 
        hue='quality', 
        col='set',
        dodge=False, 
        palette=custom_palette,
        height=4,
        aspect=1.5,
        legend_out=False,
    ) 

    g.set_xlabels("")
    g.set_ylabels("Success rate")
    legend = g._legend
    legend.set_title('')
    legend.set_frame_on(False)
    g.set_xticklabels(rotation=0)

    n_metrics = df['metric'].nunique()
    unique_methods = df['method'].unique()
    unique_metrics = df['metric'].unique()

    # Loop over each subplot
    for ax in g.axes.flatten():
        xticks = ax.get_xticks()
        xtick_labels_new = []

        # Label only center bar (e.g., Top-5 at position 1 of 3)
        for i in range(len(xticks)):
            if (i % n_metrics) == 1:
                method = unique_methods[i // n_metrics]
                metric = unique_metrics[i % n_metrics]
                # Bold method, normal metric
                label = f"$\\mathbf{{{method}}}$\n{metric}"
                xtick_labels_new.append(label)
            else:
                xtick_labels_new.append("" + "\n" + unique_metrics[i % n_metrics])

        ax.set_xticklabels(xtick_labels_new)

    for ax in g.axes.flat:
        old_title = ax.get_title()
        # old_title will be like 'time = Lunch'
        new_title = old_title.split(' = ')[-1]
        ax.set_title(new_title)
    
    #plt.ylim(0, 1.0)
    plt.savefig(f'success_rate.png', dpi=300, bbox_inches='tight')

def plot_top1_success_rate(df):
    # Define custom colors for pastel red, muted red, and dark red
    custom_palette = ["#FFC3A0", "#FF5733", "#8B0000"]

    g = sns.catplot(
        data=df, 
        kind='bar', 
        x='method', 
        y='success_rate', 
        hue='quality', 
        dodge=False, 
        palette=custom_palette,
        height=4,
        aspect=2,
        legend_out=False,
    ) 

    g.set_xlabels("")
    legend = g._legend
    legend.set_title('')
    legend.set_frame_on(False)
    g.set_xticklabels(rotation=45)

    #plt.ylim(0, 1.0)
    plt.savefig(f'top1_success_rate.png', dpi=300, bbox_inches='tight')


df = pd.read_csv('csv_files/db5_consolidated_data.csv')

plot_success(df)
