import pandas as pd
import matplotlib.pyplot as plt

# Initialize a figure
plt.figure(figsize=(10, 6))

# Load each CSV file and plot
num_files = 4
ab_loss = []
bob_loss = []
eve_loss = []

# Initialize min and max values for each set of loss data
min_ab, max_ab = None, None
min_bob, max_bob = None, None
min_eve, max_eve = None, None

# You only need to define x once if it's the same for all datasets
x = list(range(0, 2559))

for i in range(1, num_files + 1):
    df = pd.read_csv(f'test-{i}.csv')

    # ABloss
    y_ab = df['ABloss']
    ab_loss.append(y_ab)
    if min_ab is None:
        min_ab = y_ab
        max_ab = y_ab
    else:
        min_ab = pd.concat([min_ab, y_ab], axis=1).min(axis=1)
        max_ab = pd.concat([max_ab, y_ab], axis=1).max(axis=1)

    # Bobloss
    y_bob = df['Bobloss']
    bob_loss.append(y_bob)
    if min_bob is None:
        min_bob = y_bob
        max_bob = y_bob
    else:
        min_bob = pd.concat([min_bob, y_bob], axis=1).min(axis=1)
        max_bob = pd.concat([max_bob, y_bob], axis=1).max(axis=1)

    # Eveloss
    y_eve = df['Eveloss']
    eve_loss.append(y_eve)
    if min_eve is None:
        min_eve = y_eve
        max_eve = y_eve
    else:
        min_eve = pd.concat([min_eve, y_eve], axis=1).min(axis=1)
        max_eve = pd.concat([max_eve, y_eve], axis=1).max(axis=1)

# Plot the filled areas between the min and max values for each loss type
plt.fill_between(x, min_ab, max_ab, color='skyblue', alpha=0.5)
plt.fill_between(x, min_bob, max_bob, color='lightgreen', alpha=0.5)
plt.fill_between(x, min_eve, max_eve, color='salmon', alpha=0.5)

# Calculate and plot the average lines
average_ab = pd.concat(ab_loss, axis=1).mean(axis=1)
average_bob = pd.concat(bob_loss, axis=1).mean(axis=1)
average_eve = pd.concat(eve_loss, axis=1).mean(axis=1)

plt.plot(x, average_ab, color='blue', linewidth=2, label='Average AB')
plt.plot(x, average_bob, color='green', linewidth=2, label='Average Bob')
plt.plot(x, average_eve, color='red', linewidth=2, label='Average Eve')

# Customization and saving the figure
plt.xlabel('X-axis label')  # Replace with your actual label
plt.ylabel('Y-axis label')  # Replace with your actual label
plt.title('Combined Plots with Average Lines')
plt.legend()
plt.show()
