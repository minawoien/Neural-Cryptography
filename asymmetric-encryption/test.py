import pandas as pd
import matplotlib.pyplot as plt

x = list(range(0, 2500))

plt.figure(figsize=(7, 4))
for i in range(1, 5):
    pf = pd.read_csv(f'test{i}.csv')
    dictlist = pf.to_dict('list')

    pf2 = pd.read_csv(f'test{i+1}.csv')
    dictlist2 = pf2.to_dict('list')
    plt.fill_between(x, dictlist["ABloss"],
                     dictlist2["ABloss"], color='blue', alpha=0.5)
    plt.fill_between(x, dictlist["Bobloss"],
                     dictlist2["Bobloss"], color='red', alpha=0.5)
    plt.fill_between(x, dictlist["Eveloss"],
                     dictlist2["Eveloss"], color='green', alpha=0.5)
plt.xlabel("Iterations", fontsize=13)
plt.ylabel("Loss", fontsize=13)
plt.legend(fontsize=13)

plt.show()
