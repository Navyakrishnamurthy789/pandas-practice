import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset
df = pd.read_csv("titanic.csv")  

# --- 1. Survival Count ---
df['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title("Survival Count (0 = Not Survived, 1 = Survived)")
plt.xlabel("Survived")
plt.ylabel("Count")
plt.show()

# --- 2. Age Distribution ---
plt.hist(df['Age'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# --- 3. Passenger Class Pie Chart ---
df['Pclass'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap="Set3")
plt.title("Passenger Class Distribution")
plt.ylabel("")
plt.show()

# --- 4. Survival by Gender ---
df.groupby('Sex')['Survived'].mean().plot(kind='bar', color=['pink', 'lightblue'])
plt.title("Survival Rate by Gender")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.show()

# --- 5. Age vs Fare Scatter Plot ---
plt.scatter(df['Age'], df['Fare'], alpha=0.5, color='purple')
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()
