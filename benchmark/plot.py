import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})

def plot_success_rate(df):
    # Define custom colors for pastel red, muted red, and dark red
    custom_palette = ["#FFC3A0", "#FF5733", "#8B0000"]

    g = sns.catplot(
        data=df, 
        kind='bar', 
        x='method', 
        y='success_rate', 
        hue='quality', 
        col='metric',
        dodge=False, 
        palette=custom_palette,
        height=6,
        aspect=1,
        legend_out=False,
    ) 

    g.set_xlabels("")
    legend = g._legend
    legend.set_title('')
    legend.set_frame_on(False)
    g.set_xticklabels(rotation=45)

    for ax in g.axes.flat:
        old_title = ax.get_title()
        # old_title will be like 'time = Lunch'
        new_title = old_title.split(' = ')[-1]
        ax.set_title(new_title)
    
    plt.ylim(0, 1.0)
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
        height=5,
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

# List of file paths
#test_set = 'db5_test'
test_set = 'db5_ab_ag'
#csv_files = [f'{test_set}_result_diffdock_pp.csv', f'{test_set}_result_dfmdock.csv', f'{test_set}_result_dfmdock_msa.csv', f'{test_set}_result_dfmdock_interface.csv', f'{test_set}_result_afm.csv']
csv_files = [f'{test_set}_result_diffdock_pp.csv', f'{test_set}_result_dfmdock.csv',f'{test_set}_result_dfmdock_interface.csv', f'{test_set}_result_afm.csv']
#csv_files = [f'{test_set}_result_diffdock_pp.csv', f'{test_set}_result_dfmdock.csv',f'{test_set}_result_dfmdock_msa.csv', f'{test_set}_result_afm.csv']
#csv_files = [f'{test_set}_result_diffdock_pp.csv', f'{test_set}_result_dfmdock.csv', f'{test_set}_result_afm.csv']
#csv_files = [f'{test_set}_result_diffdock_pp.csv', f'{test_set}_result_afm.csv']
#csv_files = [f'{test_set}_result_4_interface.csv', f'{test_set}_result_all_interface.csv']

# Read and concatenate them
df_list = [pd.read_csv(f) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

print(df)

plot_success_rate(df)

top1_df = df[df['metric'] == 'Top-1']

plot_top1_success_rate(top1_df)



