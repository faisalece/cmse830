import streamlit as st
import pandas as pd
import seaborn as sns
import math  

import matplotlib.pyplot as plt

# dataset
st.header('Application : HW 3 (CMSE 830)')

# List of dataset filenames
data = ["water_potability.csv", "stroke-data.csv", "Life_Expectancy.csv"]

# User selects a dataset
selected_dataset = st.selectbox('Select a dataset for EDA analysis', data)

# Load the selected dataset into a DataFrame
if selected_dataset:
    df = pd.read_csv(selected_dataset)
    if selected_dataset == "water_potability.csv":
        df_name = "Water Quality Dataset"
    elif selected_dataset == "stroke-data.csv":
        df_name = "Stroke Prediction Dataset"
    elif selected_dataset == "Life_Expectancy.csv":
        df_name = "Life Expectancy (WHO) Dataset"
    else:
        df_name = "Dataset not selected"
        
    st.header(f'Name of the dataset : {df_name}')
          
    st.write("Displaying the head (first 5 rows) of the dataset :")
    st.write(df.head())  # Display the DataFrame
    st.write("Displaying the statistics of the dataset :")  # Display the DataFrame
    st.write(df.describe())  # Display the statistics
    
    # Select only numeric coloumn 
    df_new = df.select_dtypes(include=['int64', 'float64'])
    num_columns = len(df_new)
    # Define the layout of subplots
    n_rows = math.ceil(df_new.shape[1]/5)
    n_cols = 5
    
    
    #Corr Heat Map
    # Create a button to show/hide the KDE
    hide_corr = st.checkbox("Hide Correlation Heat Map")
    if hide_corr==False:
        plt.figure(figsize=(10, 10))
        sns.heatmap(df_new.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        heatmap_fig = plt.gcf()  # Get the current figure
        st.pyplot(heatmap_fig)
    
    # regression plot
    st.subheader("Regression Plot")
    plt.figure(figsize=(10, 6))
    x_column = st.selectbox('Select x column for plotting', df_new.columns)
    y_column = st.selectbox('Select y column for plotting', df_new.columns)
    plt.title(f'Regplot of {x_column} vs {y_column} with Lowess Smoother')
    fig = sns.regplot(data=df_new, x=x_column, y=y_column, lowess=True, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
    reg_fig = plt.gcf()  # Get the current figure
    st.pyplot(reg_fig)
    
    # KDE PLOT
    # Create a button to show/hide the KDE
    hide_KDE = st.checkbox("Hide KDE Plot")
    if hide_KDE==False:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 8))
        # Flatten the axes array for easier iteration
        axes = axes.flatten()
        for i, column in enumerate(df_new):
            # Select the current axis
            ax = axes[i]
            df[column].plot.kde(ax=ax)
            ax.set_title(f'KDE Plot for {column}')
        # Remove empty subplots
        for i in range(num_columns, n_rows * n_cols):
            fig.delaxes(axes[i])
        # Adjust layout
        plt.tight_layout()
        st.pyplot(fig)
    
    
    #histograms plots
    # Create a button to show/hide the histogram
    hide_histogram = st.checkbox("Hide Histogram")
    if hide_histogram==False:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
        # Flatten the axes array for easier iteration
        axes = axes.flatten()
        for i, column in enumerate(df_new):
            # Select the current axis
            ax = axes[i]
            df[column].plot.hist(ax=ax, bins=10, color='red', edgecolor='black')
            ax.set_title(f'Histogram for {column}')
        # Remove empty subplots
        for i in range(num_columns, n_rows * n_cols):
            fig.delaxes(axes[i])
        # Adjust layout
        plt.tight_layout()
        # Set the title
        plt.suptitle(f'{df_name}', y=1.03)    
        # Show the plot
        st.pyplot(fig)

