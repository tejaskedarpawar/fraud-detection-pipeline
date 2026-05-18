import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Makes plots look clean
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120

df = pd.read_csv('creditcard.csv')  # make sure the file is in the same folder

print("Shape:", df.shape)
print("\nFirst 5 rows:")
df.head()

print("Missing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nBasic stats:")
df.describe()

class_counts = df['Class'].value_counts()
fraud_pct = class_counts[1] / len(df) * 100

print(class_counts)
print(f"\nFraud is only {fraud_pct:.4f}% of all transactions!")

plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Class', data=df, palette=['steelblue', 'crimson'])
ax.set_title('Class Distribution — Legit vs Fraud', fontsize=14)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Legit (0)', 'Fraud (1)'])
ax.set_ylabel('Count')

for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df[df['Class'] == 0]['Amount'].hist(bins=50, ax=axes[0],
    color='steelblue', alpha=0.8, edgecolor='white')
axes[0].set_title('Legit Transaction Amounts')
axes[0].set_xlabel('Amount (€)')
axes[0].set_yscale('log')  # log scale because of the huge count difference

df[df['Class'] == 1]['Amount'].hist(bins=50, ax=axes[1],
    color='crimson', alpha=0.8, edgecolor='white')
axes[1].set_title('Fraud Transaction Amounts')
axes[1].set_xlabel('Amount (€)')

plt.tight_layout()
plt.show()

legit_mean = df[df['Class'] == 0]['Amount'].mean()
fraud_mean = df[df['Class'] == 1]['Amount'].mean()
print(f"Avg legit amount:  €{legit_mean:.2f}")
print(f"Avg fraud amount:  €{fraud_mean:.2f}")


fig, axes = plt.subplots(2, 1, figsize=(14, 7))

df[df['Class'] == 0]['Time'].plot(kind='hist', bins=48, ax=axes[0],
    color='steelblue', alpha=0.8, title='Legit: When do transactions happen?')
axes[0].set_xlabel('Time (seconds from first transaction)')

df[df['Class'] == 1]['Time'].plot(kind='hist', bins=48, ax=axes[1],
    color='crimson', alpha=0.8, title='Fraud: When do they happen?')
axes[1].set_xlabel('Time (seconds from first transaction)')

plt.tight_layout()
plt.show()

# Mean value of each feature split by class
fraud_mean   = df[df['Class'] == 1].mean()
legit_mean   = df[df['Class'] == 0].mean()
diff         = abs(fraud_mean - legit_mean)
top_features = diff.nlargest(12).index.tolist()

print("Features with biggest difference between fraud and legit:")
print(top_features)

# Plot distributions of top 6 V features
v_features = [f for f in top_features if f.startswith('V')][:6]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for i, feat in enumerate(v_features):
    axes[i].hist(df[df['Class']==0][feat], bins=60,
                 color='steelblue', alpha=0.6, label='Legit', density=True)
    axes[i].hist(df[df['Class']==1][feat], bins=60,
                 color='crimson', alpha=0.6, label='Fraud', density=True)
    axes[i].set_title(f'{feat} distribution')
    axes[i].legend()

plt.suptitle('Feature Distributions: Fraud vs Legit', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # hide upper triangle
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
            annot=False, linewidths=0.3, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.show()


scaler = StandardScaler()

df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_time']   = scaler.fit_transform(df[['Time']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # preserves the fraud ratio in both sets
)

print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")
print(f"Fraud cases in train: {y_train.sum()}")
print(f"Fraud cases in test:  {y_test.sum()}")

# Save for Phase 2
import joblib
joblib.dump((X_train, X_test, y_train, y_test), 'phase1_data.pkl')
print("\nData saved to phase1_data.pkl — ready for Phase 2!")
