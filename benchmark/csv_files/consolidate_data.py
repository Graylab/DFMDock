import pandas as pd

csvs = ['db5_all_result_diffdock_pp', 'db5_all_result_dfmdock', 'db5_all_result_afm', 'db5_all_result_boltz', 'db5_all_result_boltz_no_msa', 
            'db5_ab_ag_result_diffdock_pp', 'db5_ab_ag_result_dfmdock', 'db5_ab_ag_result_afm', 'db5_ab_ag_result_boltz', 'db5_ab_ag_result_boltz_no_msa',
            'db5_non_ab_ag_result_diffdock_pp', 'db5_non_ab_ag_result_dfmdock', 'db5_non_ab_ag_result_afm', 'db5_non_ab_ag_result_boltz', 'db5_non_ab_ag_result_boltz_no_msa']
labels = ['All', 'All', 'All', 'All', 'All', 'Ab-Ag', 'Ab-Ag', 'Ab-Ag', 'Ab-Ag', 'Ab-Ag', 'Non Ab-Ag', 'Non Ab-Ag', 'Non Ab-Ag', 'Non Ab-Ag', 'Non Ab-Ag']

dfs = []
for label, f in zip(labels, csvs):
    df = pd.read_csv(f'{f}.csv')
    df['set'] = label
    dfs.append(df)

df_merged = pd.concat(dfs, ignore_index=True)
df_merged.to_csv('consolidated_data.csv', index=False)
print(df_merged)
