import pandas as pd
num_files = 5
curve = 'secp384r1'
cycles = 2

loss = {'ABloss': 0, 'Bobloss': 0, 'Eveloss': 0}

for i in range(1, num_files + 1):
    df = pd.read_csv(f'{curve}/{cycles}cycle/test-{i}.csv')
    for j in loss.keys():
        y_ab = df[j][len(df[j])-1]
        loss[j] += y_ab

for j in loss.keys():
    loss[j] = round(loss[j]/num_files,3)

print(f"{loss['ABloss']} & {loss['Bobloss']} & {loss['Eveloss']}")
    