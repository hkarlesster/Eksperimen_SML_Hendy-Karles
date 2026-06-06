import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import KNNImputer
import missingno as msno
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from warnings import filterwarnings
warnings.filterwarnings("ignore")

df = pd.read_csv("heart_disease.csv")
df.head()

df.shape

df.info()

cat_cols = df.select_dtypes(include=["object"]).columns
print("Categorical Columns:", cat_cols)

for col in cat_cols:
    print(f"{col}: {df[col].unique()}")

fig, axes = plt.subplots(4, 3, figsize=(20, 20))
fig.tight_layout(pad=5.0)
axes = axes.flatten()

for i, col in enumerate(cat_cols[:12]):
    sns.countplot(x=col, data=df, hue=col, palette='Greens', dodge=False, ax=axes[i])
    axes[i].set_title(f'Plot Hitungan dari {col}')
    axes[i].tick_params(axis='x', rotation=45)

# Sembunyikan subplot yang tidak digunakan
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.show()

num_cols = df.select_dtypes(include=[ "number" ]).columns
plt.figure(figsize=( 15 ,  5 ))
df[num_cols].boxplot()
plt.xticks(rotation= 45 )
plt.show()

df.isna(). sum ()

"""Periksa apakah ada duplikat"""
df.duplicated(). sum ()

"""Mengkodekan 12 variabel kategorikal (tipe objek) sebagai variabel numerik."""
from sklearn.preprocessing import LabelEncoder

categorical_cols = df.select_dtypes(include=['object']).columns

encoder = LabelEncoder()
label_mappings = {}

for col in categorical_cols:
    mask = df[col].notna()

    df.loc[mask, col] = encoder.fit_transform(df.loc[mask, col])

    label_mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

print("Data info:")
print(df.info())

# Display label mappings
for col, mapping in label_mappings.items():
    print(f"Column: {col}")
    for label, code in mapping.items():
        print(f"{code} -> {label}")
    print()

"""Menangani data yang hilang menggunakan KNNImputer dari Scikit-learn. KNNImputer (n_neighbors=5) mengganti nilai yang hilang dengan menemukan lima tetangga terdekat untuk setiap entri yang hilang dan mengisi nilai yang hilang berdasarkan rata-ratanya."""
knn_imputer = KNNImputer(n_neighbors=5)

df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)
print(df_imputed)
df = df_imputed

"""Membuat heatmap untuk memvisualisasikan korelasi antara fitur numerik dalam dataset. Ini membantu mengidentifikasi hubungan antar variabel, tetapi dalam kasus ini, tidak ada korelasi signifikan antara fitur-fitur tersebut."""
matriks_korelasi = df.corr()

plt.figure(figsize=(12, 6))
sns.heatmap(matriks_korelasi, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Peta Panas Korelasi Fitur")
plt.show()

"""Menskalakan semua fitur numerik ke rentang [0, 1] menggunakan MinMaxScaler untuk memastikan kontribusi yang sama terhadap pelatihan model. Variabel target (Status Penyakit Jantung) dipertahankan tanpa penskalaan karena merupakan label klasifikasi. Fitur yang telah diskalakan kemudian digabungkan kembali dengan kolom target asli untuk analisis lebih lanjut."""
scaler = MinMaxScaler()

Grade_column = df[ 'Heart Disease Status' ]
df_scaled = scaler.fit_transform(df)
df = pd.DataFrame(df_scaled, columns=df.columns)

df[ 'Heart Disease Status' ] = Grade_column

df

"""Buat diagram batang untuk menggambarkan distribusi variabel target Status Penyakit Jantung."""
print(df["Heart Disease Status"].value_counts())

sns.countplot(x=df["Heart Disease Status"])
plt.title("Target Variable Distribution")
plt.show()

from sklearn.model_selection import train_test_split

# Pisahkan kelas mayoritas dan minoritas
df_majority = df[df[ 'Heart Disease Status' ] ==  0 ]
df_minority = df[df[ 'Heart Disease Status' ] ==  1 ]

# Tingkatkan jumlah sampel kelas minoritas
df_minority_upsampled = df_minority.sample(n= int ( len (df_majority) *  0.5 ), replace= True , random_state= 42 )

# Gabungkan keduanya
df_balanced = pd.concat([df_majority, df_minority_upsampled ])

# Acak dataset
df_balanced = df_balanced.sample(frac= 1 , random_state= 42 )

# Pisahkan menjadi fitur dan label
X = df_balanced.drop(columns=[ 'Heart Disease Status' ])
y = df_balanced[ 'Heart Disease Status' ]

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size= 0.2 , random_state= 42 )

# Menyimpan data preprocessing ke dalam csv
df.to_csv('heart_disease_preprocessed.csv', index=False)
print('Data berhasil disimpan !')