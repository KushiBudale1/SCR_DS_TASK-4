'''Analyze traffic accident data to identify patterns related to road conditions, weather, and time of day. 
Visualize accident hotspots and contributing factors.'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load a sample of the dataset for faster processing
data_filepath = "US_Accidents_Dec21_updated.csv"
df = pd.read_csv(data_filepath, nrows=100000)  
print("Data loaded successfully.")
print(df.head(10))

# General information about the dataset
print("Number of columns:", len(df.columns))
print("Number of rows:", len(df))
df.info()

# Drop columns with high percentage of missing values
missing_threshold = 0.3
df = df.dropna(thresh=len(df) * (1 - missing_threshold), axis=1)
print("Dropped columns with more than 30% missing values.")

# Separate categorical and numerical features
df_cat = df.select_dtypes('object').drop(['ID'], axis=1, errors='ignore')
df_num = df.select_dtypes(include=np.number)

# Drop columns that are not useful for analysis
df_cat.drop(['Description', 'Zipcode', 'Weather_Timestamp'], axis=1, inplace=True, errors='ignore')
df.drop(['Airport_Code'], axis=1, inplace=True, errors='ignore')

# Convert 'Start_Lng' and 'Start_Lat' to numeric
df['Start_Lng'] = pd.to_numeric(df['Start_Lng'], errors='coerce')
df['Start_Lat'] = pd.to_numeric(df['Start_Lat'], errors='coerce')
df['Temperature(F)'] = pd.to_numeric(df['Temperature(F)'], errors='coerce')

# Plot correlation heatmap of numerical features with annotations
plt.figure(figsize=(10, 8))
sns.heatmap(df_num.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Top 10 cities by number of accidents
top_cities = df['City'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_cities.values, y=top_cities.index, palette='viridis')
plt.title('Top 10 Cities By Number of Accidents')
plt.xlabel('Number of Accidents')
plt.show()

# Accident severity distribution
plt.figure(figsize=(6, 6))
severity_counts = df['Severity'].value_counts()
plt.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%', 
        colors=sns.color_palette('pastel'))
plt.title('Accidents by Severity')
plt.show()

# Plot accidents by time of day
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['Hour'] = df['Start_Time'].dt.hour
plt.figure(figsize=(10, 6))
sns.histplot(df['Hour'], bins=24, kde=False, color='dodgerblue')
plt.xlabel('Hour of Day')
plt.ylabel('Accident Count')
plt.title('Accidents Count by Time of Day')
plt.tight_layout()
plt.show()

# Weather conditions bar plot (Top 10)
top_weather_conditions = df['Weather_Condition'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=top_weather_conditions.values, y=top_weather_conditions.index, palette='magma')
plt.title('Top 10 Weather Conditions at Time of Accident')
plt.xlabel('Number of Accidents')
plt.show()

# Scatter plot of accident locations with state labels
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Start_Lng', y='Start_Lat', hue='State', palette='hsv', alpha=0.5, legend=False, data=df)
plt.title('Accidents Locations by State')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Add state labels at the mean position of accidents
for state in df['State'].unique():
    state_data = df[df['State'] == state]
    plt.text(state_data['Start_Lng'].mean(), state_data['Start_Lat'].mean(), state,
             horizontalalignment='center', size='medium', color='black', weight='semibold')

plt.tight_layout()
plt.show()

# Box Plot of Temperature by Accident Severity
plt.figure(figsize=(12, 8))
sns.boxplot(x='Severity', y='Temperature(F)', data=df)
plt.title('Box Plot of Temperature by Accident Severity')
plt.xlabel('Severity')
plt.ylabel('Temperature (F)')
plt.show()

# Violin Plot of Wind Speed by Accident Severity
plt.figure(figsize=(12, 8))
sns.violinplot(x='Severity', y='Wind_Speed(mph)', data=df)
plt.title('Violin Plot of Wind Speed by Accident Severity')
plt.xlabel('Severity')
plt.ylabel('Wind Speed (mph)')
plt.show()

# Density Plot of Temperature
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Temperature(F)'].dropna(), fill=True, color='skyblue')
plt.title('Density Plot of Temperature')
plt.xlabel('Temperature (F)')
plt.ylabel('Density')
plt.show()

# FacetGrid of Temperature by Severity
g = sns.FacetGrid(df, col='Severity', col_wrap=3, height=4, aspect=1.5)
g.map(sns.histplot, 'Temperature(F)')
g.set_axis_labels('Temperature (F)', 'Count')
g.set_titles('Severity: {col_name}')
plt.show()

# Pairplot with adjusted height, aspect, marker size, and transparency
df_sample = df.sample(n=5000, random_state=42)
pairplot = sns.pairplot(df_sample[df_num.columns], diag_kind='kde', corner=True, 
                        height=2.5, aspect=1.0, plot_kws={'s': 25, 'alpha': 0.6})
plt.suptitle('Pairplot of Sampled Numerical Features', y=1.0, fontsize=24, weight='bold')
pairplot.fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)
# Ensure pairplot.axes.flat is valid
for ax in pairplot.axes.flat:
    if ax is not None:
        ax.set_ylabel(ax.get_ylabel(), rotation=0, labelpad=30)
plt.show()

# Bubble Plot of Accident Locations with Temperature
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Start_Lng', y='Start_Lat', size='Temperature(F)', sizes=(20, 200), 
                hue='Severity', palette='coolwarm', alpha=0.5, data=df)
plt.title('Bubble Plot of Accident Locations with Temperature')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


