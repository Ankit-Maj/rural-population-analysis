# =============================================================================
# Preliminary Setup: Data Loading & Cleaning
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

file_path = "./rural_population.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

print("📊 Dataset Overview")
print("Shape of data:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

df_clean = df.dropna(subset=["Area Name"])
df_clean_no_national = df_clean[~df_clean["Level"].isin(["INDIA"])].copy()

village_columns = [col for col in df.columns if "Number of Villages" in col]
df_clean_no_national[village_columns] = df_clean_no_national[village_columns].fillna(0)

# =============================================================================
# Objective 1: Summarize Rural Population Distribution & Infrastructure
# =============================================================================
plt.figure(figsize=(12, 7))
subdistrict_data = df_clean_no_national[df_clean_no_national["Level"] == "SUB-DISTRICT"]

ax1 = sns.histplot(subdistrict_data["Total Rural population - Persons"], 
                   bins=20, 
                   kde=True,
                   color='steelblue',
                   edgecolor='white',
                   linewidth=1,
                   alpha=0.7)

plt.title("Distribution of Rural Population at Sub-District Level", fontsize=14, pad=20)
plt.xlabel("Population (Persons)", fontsize=12, labelpad=10)
plt.ylabel("Number of Districts", fontsize=12, labelpad=10)
plt.xscale('log')

ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax1.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))

plt.grid(axis='y', alpha=0.3)

stats_text = (f"Summary Statistics:\n"
              f"Count: {len(subdistrict_data):,}\n"
              f"Mean: {subdistrict_data['Total Rural population - Persons'].mean():,.0f}\n"
              f"Median: {subdistrict_data['Total Rural population - Persons'].median():,.0f}")
plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
             fontsize=10, va='top')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
level_counts = df_clean_no_national["Level"].value_counts()

plt.pie(level_counts, labels=level_counts.index, autopct='%1.1f%%', 
        startangle=90, shadow=False, explode=[0.05] * len(level_counts),
        colors=sns.color_palette('viridis', len(level_counts)))

centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')  
plt.title("Distribution of Administrative Levels", fontsize=14, pad=20)

labels = [f"{label} ({count})" for label, count in zip(level_counts.index, level_counts)]
plt.legend(labels, loc="best", bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()

# =============================================================================
# Objective 2: Categorize Villages by Population Size Brackets
# =============================================================================
village_brackets = [
    "Less than 200 - Number of Villages",
    "200-499 - Number of Villages",
    "500-999 - Number of Villages",
    "1000-1999 - Number of Villages",
    "2000-4999 - Number of Villages",
    "5000-9999 - Number of Villages",
    "10000 and above - Number of Villages"
]

village_by_state = df_clean_no_national[df_clean_no_national["Level"] == "STATE"]\
                     .groupby("Area Name")[village_brackets].sum()

ax_stack = village_by_state.plot(kind="bar", stacked=True, figsize=(14,8), colormap="viridis")
plt.title("Village Size Distribution by State")
plt.xlabel("State")
plt.ylabel("Number of Villages")
plt.xticks(rotation=90)
plt.legend(title="Village Size", loc="upper left")
plt.tight_layout()

for p in ax_stack.patches:
    width = p.get_width()
    height = p.get_height()
    if height > 0:
        x_center = p.get_x() + width/2
        y_center = p.get_y() + height/2
        ax_stack.text(x_center, y_center, f'{int(height)}', ha='center', va='center', fontsize=7, color='white')

plt.show()

# =============================================================================
# Objective 3: Identify Regions with Varying Settlement Density
# =============================================================================
df_subdistrict = df_clean[df_clean["Level"] == "SUB-DISTRICT"].copy()

df_subdistrict["Village Density"] = (df_subdistrict["Total number of inhabited villages"] / 
                                       df_subdistrict["Total Rural population - Persons"]) * 1000

highest_density = df_subdistrict.sort_values(by="Village Density", ascending=False).head(5)
lowest_density = df_subdistrict.sort_values(by="Village Density", ascending=True).head(5)

print("\n🔍 Top 5 Sub-districts with Highest Village Density (villages per 1000 persons):")
print(highest_density[["Area Name", "Village Density", "Total Rural population - Persons"]])

print("\n🔍 Top 5 Sub-districts with Lowest Village Density (villages per 1000 persons):")
print(lowest_density[["Area Name", "Village Density", "Total Rural population - Persons"]])

# =============================================================================
# Objective 4: Evaluate Regional Gender Ratios
# =============================================================================
df_clean_no_national["Sex Ratio"] = (df_clean_no_national["Total Rural population - Females"] / 
                                     df_clean_no_national["Total Rural population - Males"]) * 1000

sex_ratio_by_state = df_clean_no_national[df_clean_no_national["Level"] == "STATE"]\
                     .groupby("Area Name")["Sex Ratio"].mean().sort_values()

plt.figure()
ax4 = sex_ratio_by_state.plot(kind="barh", color="teal")
plt.title("Average Rural Sex Ratio by State\n(Females per 1000 Males)")
plt.xlabel("Sex Ratio (Females per 1000 Males)")
plt.ylabel("State")

for i, (value, state) in enumerate(zip(sex_ratio_by_state, sex_ratio_by_state.index)):
    plt.text(value + 5, i, f"{value:.0f}", va='center', fontsize=8)

plt.tight_layout()
plt.show()

# =============================================================================
# Objective 5: Compare National and Regional Demographic Trends
# =============================================================================
df_national = df[df["Level"] == "INDIA"].copy()
df_state = df[df["Level"] == "STATE"].copy()
df_district = df[df["Level"] == "DISTRICT"].copy()

indicators = ["Total Rural population - Persons", "Total Rural population - Males", 
              "Total Rural population - Females", "Total number of inhabited villages"]

print("\n📊 National-Level Summary:")
print(df_national[["Area Name"] + indicators].set_index("Area Name"))

print("\n📊 State-Level Summary Statistics (mean, min, max):")
state_summary = df_state[indicators].agg(["mean", "min", "max"]).T
print(state_summary)

if not df_district.empty:
    print("\n📊 District-Level Summary Statistics (mean, min, max):")
    district_summary = df_district[indicators].agg(["mean", "min", "max"]).T
    print(district_summary)
else:
    print("\nℹ️ No district-level data available in the dataset.")

# =============================================================================
# Additional: Correlation Analysis
# =============================================================================
# Select relevant numerical columns for correlation analysis
numeric_cols = [
    "Total Rural population - Persons", 
    "Total Rural population - Males", 
    "Total Rural population - Females",
    "Total number of inhabited villages",
    "Sex Ratio"
]

# Calculate correlation matrix
corr_matrix = df_clean_no_national[numeric_cols].corr()

# Create heatmap visualization
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", 
            linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix of Key Rural Population Metrics", fontsize=14, pad=20)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# =============================================================================
# Additional: Outlier Detection on Total Rural Population - Persons
# =============================================================================
pop_col = "Total Rural population - Persons"

q1 = df_clean_no_national[pop_col].quantile(0.25)
q3 = df_clean_no_national[pop_col].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers_iqr = df_clean_no_national[(df_clean_no_national[pop_col] < lower_bound) | (df_clean_no_national[pop_col] > upper_bound)]
print(f"\n🔍 IQR Method: Found {outliers_iqr.shape[0]} outliers in '{pop_col}'")
print(outliers_iqr[["Area Name", pop_col]])

z_scores = np.abs(stats.zscore(df_clean_no_national[pop_col]))
threshold_z = 3
outliers_z = df_clean_no_national[z_scores > threshold_z]
print(f"\n🔍 Z-score Method: Found {outliers_z.shape[0]} outliers in '{pop_col}' (|z| > {threshold_z})")
print(outliers_z[["Area Name", pop_col]])

alpha = 0.05
n = len(df_clean_no_national[pop_col])
t_threshold = stats.t.ppf(1 - alpha/2, df=n-1)
t_scores = np.abs((df_clean_no_national[pop_col] - df_clean_no_national[pop_col].mean()) / df_clean_no_national[pop_col].std())
outliers_t = df_clean_no_national[t_scores > t_threshold]
print(f"\n🔍 T-test Method: Found {outliers_t.shape[0]} outliers in '{pop_col}' (|t| > {t_threshold:.2f})")
print(outliers_t[["Area Name", pop_col]])