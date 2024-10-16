from pinder.core.index.utils import get_index


pindex = get_index()
testset = pindex.query(f'pinder_s == True')
df = testset[['id', 'holo_R_pdb', 'holo_L_pdb']]


# Define the path string to prepend
path_string = "/scratch16/jgray21/lchu11/data/pinder/2024-02/test_set_pdbs/"

# Add the path string to column2 and column3
df['holo_R_pdb'] = path_string + df['holo_R_pdb'].astype(str)
df['holo_L_pdb'] = path_string + df['holo_L_pdb'].astype(str)

# Display the updated DataFrame
print(df)

# Save the DataFrame to a CSV file without header and index
df.to_csv('pinder_s.csv', header=False, index=False)
