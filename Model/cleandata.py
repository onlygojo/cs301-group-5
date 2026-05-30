import pandas as p


def clean_data():
    
# Load data
    data = p.read_csv("Data/data.csv")
    data = data.drop(['id','Unnamed: 32'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

# Remove empty columns, missing values, and duplicates
# data = data.dropna(axis=1, how='all')
# data = data.dropna()
# data = data.drop_duplicates()

# # Clean data
# data = data.drop('id', axis=1)
# data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# # Explore the data
# print("DATA EXPLORATION")

# print("\nDataset Shape (rows, columns):", data.shape)

# print("\nData Types:")
# print(data.dtypes)

# print("\nFirst 5 rows:")
# print(data.head())

# print("\nBasic Statistics:")
# print(data.describe())

# print("\nMissing Values:")
# print(data.isnull().sum())

# print("\nDiagnosis Distribution:")
# print(data['diagnosis'].value_counts())

# # Display cleaned data
# print("FULL CLEANED DATA")
# print(data)