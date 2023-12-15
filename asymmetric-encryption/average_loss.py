import pandas as pd

# Define the number of files, the curve and the number of cycles
num_files = 5
curve = 'secp384r1'
cycles = 1

loss = {'ABloss': 0, 'Bobloss': 0, 'Eveloss': 0}

# Sum the last loss value of each file
for i in range(1, num_files + 1):
    df = pd.read_csv(f'{curve}/{cycles}cycle/test-{i}.csv')
    for j in loss.keys():
        y_ab = df[j][len(df[j])-1]
        loss[j] += y_ab

# Divide the sum by the number of files
for j in loss.keys():
    loss[j] = round(loss[j]/num_files,3)

# Print the average loss values
print(f"{loss['ABloss']} & {loss['Bobloss']} & {loss['Eveloss']}")
    