import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

def plot_dockq(df):
    g = sns.jointplot(
        data=df,
        x='DFMDOCK_MSA',
        y='AFM',
        kind='scatter',
        marginal_kws=dict(bins=30, fill=True)
    )
    # Plot y = x line within [0, 1]
    g.ax_joint.plot([0, 1], [0, 1], '--', color='gray', zorder=0)

    # Set axis limits
    g.ax_joint.set_xlim(0, 1)
    g.ax_joint.set_ylim(0, 1)

    plt.savefig(f'db5_test_best_dockq.png', dpi=300, bbox_inches='tight')

# List of file paths
test_set = 'db5_test'
df1 = pd.read_csv(f'{test_set}_best_dockq_dfmdock_msa.csv')
df2 = pd.read_csv(f'{test_set}_top5_dockq_afm.csv')

df1 = df1.rename(columns={'DockQ': 'DFMDOCK_MSA'})
df2 = df2.rename(columns={'DockQ': 'AFM'})

# Merge on 'id'
df = pd.merge(df1, df2, on='id')

plot_dockq(df)



