import pandas as pd

# Create a sample dataframe
Biodata = {'Name2': ['John', 'Emily', 'Mike', 'Lisa'],
           'Age': [28, 23, 35, 31],
           'Gender': ['M', 'F', 'M', 'F']
           }


df = pd.DataFrame(Biodata)

# Save the dataframe to a CSV file
df.to_csv('Biodata.csv', mode='a', index=False)
