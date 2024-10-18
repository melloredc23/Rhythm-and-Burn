#For requirments file
#!pip install -r requirements.txt
#pip install matplotlib
#pip install scikit-learn
#pip install seaborn

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
#to help download the dataset
import os
#to help open the dataset from Kaggle
import opendatasets as od
import plotly.graph_objects as go
import plotly.figure_factory as ff


#STREAMLIT APP
st.title("**Rhythm & Burn**: Exploring the Fitness Benefits of Dance Styles")
st.header("Welcome to my Streamlit app!")

#Overview
#st.markdown("This app shows if there is a correlation between the tempo and health benefits between different dance styles to determine which dance style is better for physical fitness and health.")
df = pd.read_csv('dance data.csv', encoding='latin-1')
df2 = pd.read_csv("exercise_dataset.csv")
#df = 'https://www.kaggle.com/datasets/melissamonfared/dances/discussion/518578'
#df2 = 'https://www.kaggle.com/datasets/aadhavvignesh/calories-burned-during-exercise-and-activities/discussion/144014'
d3 = df2.drop(['130 lb', '155 lb', '180 lb', '205 lb'], axis=1)
d3 = d3.iloc[[29, 34, 35, 36],:]

#Sidebar
section = st.sidebar.radio("Navigate to:", ['Introduction', 'Data Overview', 'Correlation Analysis', 'Visualizations', 'Conclusion'])
#Intro Page
if section == 'Introduction':
    st.markdown("**Introduction**")
    st.write("This project explores different dance styles from all around the world, focusing on the relationship between **tempo** and **health benefits** to identify which styles offers the greatest physical fitness benefit")
    st.write("To connect this data, I also analyzed the relationship between **tempo** and **calories** to identify if faster-tempo dances result in higher calorie expenditure")
    st.image('/Users/kendallandrews/Downloads/groupdance.jpg')
    #provide pictures of dances

#Data Overview - summarize the datasets (tables, key statistics, and visual overview - allow users to explore data through sorting and filtering)
if section == 'Data Overview':
    st.header("Data Overview")

    tab1, tab2 = st.tabs(["Dances", "Calories"])
    with tab1:
        st.header("Dance Styles and Genres")

    #Load DANCE dataset
    #dataset = 'https://www.kaggle.com/datasets/melissamonfared/dances/discussion/518578'
    # Using opendatasets let's download the data sets
    #od.download(dataset, force=True)
    # #kaggle datasets download -d melissamonfared/dances
    #data_dir = './dances'
    #os.listdir(data_dir)
    #encoding='latin-1' - helps solve the error: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf1 in position 41293: invalid continuation byte

        df = pd.read_csv('dance data.csv', encoding='latin-1')
    #df = pd.read_csv('/Users/kendallandrews/Downloads/dances/dance data.csv')

        st.write("Dance Styles and Genres Dataset")
        st.dataframe(df)

    #Summary stats
        st.subheader("Summary Statistics")
        st.write(df.describe())

    #Filter data
        filter = st.selectbox("Filter by Dance Style", df["Dance style"].unique())
        filtered = df[df["Dance style"]  == filter]
        st.write(filtered)

    #IDA Breakdown
        st.subheader("IDA Breakdown")
    #Handle missing values - Remove duplicates - Correct data types - Standardize formats
    #Missing Values - show code breakdown
        st.write("Missing Values")
        code = '''
    for i in df.isnull().sum():
    if i > 0:
        print(df.isnull().sum())
    else:
        pass
        '''
        st.code(code, language="python")

    #Duplicates - Show code breakdown
        st.write("Duplicates")
        code='''
    duplicate_rows = df.duplicated()

# print duplicate rows
for i in duplicate_rows:
    if i == True:
        print(duplicate_rows)
    else:
        pass
    '''
        st.code(code, language="python")    

    #Correct data types
        st.write("Data Types")
        st.write("The data types was mostly categorical and the main variables I worked with which were numerical or later turned numerical was **Hardness Ratio, Tempo (BPM), Learning Difficulty, Health Benefits, Age Group**")
    
        st.write("This is an example of how I changes the health benefits variable from catergorical - textual to categorical - ordinal")
        code='''
    df["Health Benefits"]
df["health_count"] = df["Health Benefits"].str.split(",").apply(len)
print(df[["Health Benefits", "health_count"]])
    '''
        st.code(code, language="python") 

    #Encoding 
        st.write("Encoding")
        code='''
    from sklearn.preprocessing import OrdinalEncoder
#LabelEncoder, OneHotEncode
oe = OrdinalEncoder(categories=[[ 'Very easy','Easy', 'Moderate', 'Hard', 'Difficult']])
df['difficulty'] = oe.fit_transform(df[['Learning Difficulty']])
print(df[['Learning Difficulty', 'difficulty']])

#ages - object (all ages, teens and young adults, Adults, children and adults) - can change to numerical (CHECK)
#not sure if age is relevant yet
df["Age Group"]
# replace error in age column to the correct label
df['Age Group'].replace('574]: All ages]', 'All ages', inplace=True)
#'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)'
oe = OrdinalEncoder(categories=[['Children','Teens and young adults', 'Children and Adults', 'Teens and young adults, Adults','Adults', 'All ages']])
df['ages'] = oe.fit_transform(df[['Age Group']])
print(df[['Age Group', 'ages']])
    
    '''
        st.code(code, language="python")

    #Standarized the data
        st.write("Standarized the data")
        code='''
    #Standardize formats
numeric_cols_df = df.select_dtypes(include=[np.number]).columns
z1= df[numeric_cols_df].apply(zscore)
print(z1)
    '''
        st.code(code, language="python")

    with tab2:
        st.header("Calories")

        #Load CALORIES dataset
        #dataset2 = 'https://www.kaggle.com/datasets/aadhavvignesh/calories-burned-during-exercise-and-activities/discussion/144014'
        #od.download(dataset2, force=True)
        #data_dir2 = './calories-burned-during-exercise-and-activities'
        #os.listdir(data_dir2)
        df2 = pd.read_csv("exercise_dataset.csv")
    
        st.write("Calories Dataset")
        st.dataframe(df2)

        #st.write("Final Calories Dataset")

        #Edited Calories Dataset
        d3 = df2.drop(['130 lb', '155 lb', '180 lb', '205 lb'], axis=1)
        d3 = d3.iloc[[29, 34, 35, 36],:]
        st.write("Final Calories Dataset")
        st.dataframe(d3)

        #Summary stats
        st.subheader("Summary Statistics")
        st.write(d3.describe())

        #Filter data
        filter2 = st.selectbox("Filter Activity", d3["Activity, Exercise or Sport (1 hour)"].unique())
        filtered2 = d3[d3["Activity, Exercise or Sport (1 hour)"]  == filter2]
        st.write(filtered2)

#Correlation Analysis
if section == 'Correlation Analysis':
    df = pd.read_csv('dance data.csv', encoding='latin-1')
    st.header("Correlation Analysis Between Tempo and Health Benefits")
    #df["Health Benefits"]
    df["health_count"] = df["Health Benefits"].str.split(",").apply(len)
    #print(df[["Health Benefits", "health_count"]])

    #learning difficulty - object (easy, moderate, hard) - can change to numerical (CHECK)
    from sklearn.preprocessing import OrdinalEncoder
    #LabelEncoder, OneHotEncode
    oe = OrdinalEncoder(categories=[[ 'Very easy','Easy', 'Moderate', 'Hard', 'Difficult']])
    df['difficulty'] = oe.fit_transform(df[['Learning Difficulty']])
#print(df[['Learning Difficulty', 'difficulty']])

#ages - object (all ages, teens and young adults, Adults, children and adults) - can change to numerical (CHECK)
#not sure if age is relevant yet
#df["Age Group"]
# replace error in age column to the correct label
    df['Age Group'].replace('574]: All ages]', 'All ages', inplace=True)
#'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)'
    oe = OrdinalEncoder(categories=[['Children','Teens and young adults', 'Children and Adults', 'Teens and young adults, Adults','Adults', 'All ages']])
    df['ages'] = oe.fit_transform(df[['Age Group']])
#print(df[['Age Group', 'ages']])

    fig, ax = plt.subplots()
    st.subheader('Scatter Plot of Tempo vs. Health Benefits')
    sns.regplot(x='Tempo (BPM)', y='health_count', data=df)
    st.pyplot()
    

#Visualizations 

#IDA Portion
#- Missing data plots - correlation heatmaps
# Outliers - bar plots
if section == 'Visualizations':
    df = pd.read_csv('dance data.csv', encoding='latin-1')
    st.header("Visualizations")

    #df["Health Benefits"]
    df["health_count"] = df["Health Benefits"].str.split(",").apply(len)
    #print(df[["Health Benefits", "health_count"]])

    #learning difficulty - object (easy, moderate, hard) - can change to numerical (CHECK)
    from sklearn.preprocessing import OrdinalEncoder
    #LabelEncoder, OneHotEncode
    oe = OrdinalEncoder(categories=[[ 'Very easy','Easy', 'Moderate', 'Hard', 'Difficult']])
    df['difficulty'] = oe.fit_transform(df[['Learning Difficulty']])
    #print(df[['Learning Difficulty', 'difficulty']])

    #ages - object (all ages, teens and young adults, Adults, children and adults) - can change to numerical (CHECK)
    #not sure if age is relevant yet
    #df["Age Group"]
    # replace error in age column to the correct label
    df['Age Group'].replace('574]: All ages]', 'All ages', inplace=True)
    #'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)'
    oe = OrdinalEncoder(categories=[['Children','Teens and young adults', 'Children and Adults', 'Teens and young adults, Adults','Adults', 'All ages']])
    df['ages'] = oe.fit_transform(df[['Age Group']])
    #print(df[['Age Group', 'ages']])


    tab1, tab2, tab3= st.tabs(["Dances", "Calories", "Health Benefits"])
    with tab1:
        st.header("Dance Styles and Genres")
        #Line chart for tempo and health benefits
        st.line_chart(df[["Tempo (BPM)", "health_count"]])


        # Calculate correlation matrix
        selected_features = ['Hardness Ratio', 'Tempo (BPM)', 'health_count', 'difficulty', 'ages']
        corr_matrix = df[selected_features].corr().values
        # Display the correlation matrix using a heatmap
        st.header('Correlation Matrix')
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, ax=ax, cmap='coolwarm')
        plt.title("Correlation Heatmap of Dance Dataset")
        plt.xlabel('Features')
        plt.ylabel('Features')
        st.pyplot(fig)
        #print visualization
        #fig_heatmap.show()

        #Missingness
        fig, ax = plt.subplots()
        sns.heatmap(df.isna(), cmap="magma")
        plt.title("Heatmap of Missingness in Dance Dataset")
        st.pyplot(fig)

        st.subheader("Outliers")

        # Create the countplot - difficulty
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Hardness Ratio', ax=ax)
        #new_labels = [0.5,1,1.5,2]
        #ax.set_xticklabels(range(0,2,1))
        plt.title("Hardness Ratio Outlier")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Tempo (BPM)', ax=ax)
        # Modify x-axis labels
        #new_labels = [50,100,150,200,250]
        #ax.set_xticklabels(ax.set_xticklabels(range(50,250,50)))
        plt.title("Tempo Outlier")
        st.pyplot(fig)

        # Create the countplot - difficulty
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='difficulty', ax=ax)
        plt.title("Difficulty Outlier")
        st.pyplot(fig)

        #health benefits
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='health_count', ax=ax)
        plt.title("Health Benefits Outlier")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.countplot(data=df, x='ages', ax=ax)
        plt.title("Ages Outlier")
        st.pyplot(fig)

    with tab2:
        st.header("Calories")
    
        #Line chart for tempo and health benefits
        st.line_chart(d3[["Activity, Exercise or Sport (1 hour)", "Calories per kg"]])

        # Calculate correlation matrix
        selected_features = ['130 lb', '155 lb', '180 lb', '205 lb', 'Calories per kg']
        corr_matrix = df2[selected_features].corr().values
        # Display the correlation matrix using a heatmap
        st.header('Correlation Matrix')
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, ax=ax, cmap='coolwarm')
        plt.title("Correlation Heatmap of Calories Dataset")
        plt.xlabel('Features')
        plt.ylabel('Features')
        st.pyplot(fig)
        #print visualization
        #fig_heatmap.show()

        #Missingness
        fig, ax = plt.subplots()
        sns.heatmap(df2.isna(), cmap="magma")
        plt.title("Heatmap of Missingness in Calories Dataset")
        st.pyplot(fig)

        st.subheader("Outliers")

        # Create the countplot - difficulty
        fig, ax = plt.subplots()
        sns.countplot(data=df2, x='130 lb', ax=ax)
        #new_labels = [0.5,1,1.5,2]
        #ax.set_xticklabels(range(0,2,1))
        plt.title("130 lb Outlier")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.countplot(data=df2, x='155 lb', ax=ax)
        # Modify x-axis labels
        #new_labels = [50,100,150,200,250]
        #ax.set_xticklabels(ax.set_xticklabels(range(50,250,50)))
        plt.title("155 lb Outlier")
        st.pyplot(fig)

        # Create the countplot - difficulty
        fig, ax = plt.subplots()
        sns.countplot(data=df2, x='180 lb', ax=ax)
        plt.title("180 lb Outlier")
        st.pyplot(fig)

        #health benefits
        fig, ax = plt.subplots()
        sns.countplot(data=df2, x='205 lb', ax=ax)
        plt.title("205 lb Outlier")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sns.countplot(data=df2, x='Calories per kg', ax=ax)
        plt.title("Calories Outlier")
        st.pyplot(fig)

    with tab3:
        st.header("Health Benefits")
        #HEALTH BENEFITS MATRIX

        slice = df['Health Benefits'].str.split(",")
        my_list = slice.tolist()
        dtype_after = type(my_list)

        #Create an empty list of each health benefit
        #If the health benefit is prevalent in that dance add a 1 to the list 'health benefit' if not add a 0
        #benefits
        dict_list = []
        Improved_cardiovascular_health = []
        for i in range(len(my_list)):
            if 'Improved cardiovascular health' in my_list[i]:
                Improved_cardiovascular_health.append(1)
                data = {'Improved cardiovascular health': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Improved cardiov' in my_list[i]:
                Improved_cardiovascular_health.append(1)
                data = {'Improved cardiovascular health': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Improved cardiovascular health and increased physical fitness' in my_list[i]:
                Improved_cardiovascular_health.append(1)
                data = {'Improved cardiovascular health': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Cardiovascular health' in my_list[i]:
                Improved_cardiovascular_health.append(1)
                data = {'Improved cardiovascular health': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Positive effects on cardiovascular health' in my_list[i]:
                Improved_cardiovascular_health.append(1)
                data = {'Improved cardiovascular health': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Moderate cardiovascular workout' in my_list[i]:
                Improved_cardiovascular_health.append(1)
                data = {'Improved cardiovascular health': df["Dance style"][i]}
                dict_list.append(data)
            elif ' such as improved cardiovascular' in my_list[i]:
                Improved_cardiovascular_health.append(1)
                data = {'Improved cardiovascular health': df["Dance style"][i]}
                dict_list.append(data)
            else:
                Improved_cardiovascular_health.append(0)
        
            #print(Improved_cardiovascular_health)

        Improved_flexibility = []
        for i in range(len(my_list)):
            if " Improved flexibility" in my_list[i]:
                Improved_flexibility.append(1)
                data = {'Improved flexibility': df["Dance style"][i]}
                dict_list.append(data)
            elif " and Improved flexibility." in my_list[i]:
                Improved_flexibility.append(1)
                data = {'Improved flexibility': df["Dance style"][i]}
                dict_list.append(data)
            elif ' and Improved flexibility' in my_list[i]:
                Improved_flexibility.append(1)
                data = {'Improved flexibility': df["Dance style"][i]}
                dict_list.append(data)
            elif " and improved flexibility" in my_list[i]:
                Improved_flexibility.append(1)
                data = {'Improved flexibility': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Improved flexibility and strength' in my_list[i]:
                Improved_flexibility.append(1)
                data = {'Improved flexibility': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Improved strength and flexibility' in my_list[i]:
                Improved_flexibility.append(1)
                data = {'Improved flexibility': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Flexibility' in my_list[i]:
                Improved_flexibility.append(1)
                data = {'Improved flexibility': df["Dance style"][i]}
                dict_list.append(data)
            else:
             Improved_flexibility.append(0)
                #print(Improved_flexibility)

        Stress_relief = []
        for i in range(len(my_list)):
            if " Stress relief" in my_list[i]:
                Stress_relief.append(1)
                data = {'Stress relief': df["Dance style"][i]}
                dict_list.append(data)
            elif "Stress relief" in my_list[i]:
                Stress_relief.append(1)
                data = {'Stress relief': df["Dance style"][i]}
                dict_list.append(data)
            elif ' stress relief' in my_list[i]:
                Stress_relief.append(1)
                data = {'Stress relief': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Reduces stress' in my_list[i]:
                Stress_relief.append(1)
                data = {'Stress relief': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Stress relief and improved mood' in my_list[i]:
                Stress_relief.append(1)
                data = {'Stress relief': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Reduces stress and anxiety'  in my_list[i]:
                Stress_relief.append(1)
                data = {'Stress relief': df["Dance style"][i]}
                dict_list.append(data)
            else:
                Stress_relief.append(0)
        #print(Stress_relief)

        #added to cerebellum
        Improved_posture = []
        for i in range(len(my_list)):
            if " Improved posture" in my_list[i]:
                Improved_posture.append(1)
                data = {'Improves posture': df["Dance style"][i]}
                dict_list.append(data)
            elif "Improved posture" in my_list[i]:
                Improved_posture.append(1)
                data = {'Improves posture': df["Dance style"][i]}
                dict_list.append(data)
            else:
                Improved_posture.append(0)
        #print(Improved_posture)

        confidence = []
        for i in range(len(my_list)):
            if " and Increased self-confidence" in my_list[i]:
                confidence.append(1)
                data = {'Confidence': df["Dance style"][i]}
                dict_list.append(data)
            elif " Increased self-confidence" in my_list[i]:
                confidence.append(1)
                data = {'Confidence': df["Dance style"][i]}
                dict_list.append(data)
            elif "Increased self-confidence" in my_list[i]:
                confidence.append(1)
                data = {'Confidence': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Improved self-confidence' in my_list[i]:
                confidence.append(1)
                data = {'Confidence': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Increased self-esteem' in my_list[i]:
                confidence.append(1)
                data = {'Confidence': df["Dance style"][i]}
                dict_list.append(data)
            elif ' and Increased self-' in my_list[i]:
                confidence.append(1)
                data = {'Confidence': df["Dance style"][i]}
                dict_list.append(data)
            else:
               confidence.append(0)
        #print(confidence)

        physical_fitness = []
        for i in range(len(my_list)):
            if " and increased physical fitness" in my_list[i]:
                physical_fitness.append(1)
                data = {'Increased physical fitness': df["Dance style"][i]}
                dict_list.append(data)
            elif " Increased physical fitness" in my_list[i]:
                physical_fitness.append(1)
                data = {'Increased physical fitness': df["Dance style"][i]}
                dict_list.append(data)
            elif "Improved physical fitness" in my_list[i]:
                physical_fitness.append(1)
                data = {'Increased physical fitness': df["Dance style"][i]}
                dict_list.append(data)
            elif " Improved physical fitness" in my_list[i]:
                physical_fitness.append(1)
                data = {'Increased physical fitness': df["Dance style"][i]}
                dict_list.append(data)
            elif '  Increased physical activity' in my_list[i]:
                physical_fitness.append(1)
                data = {'Increased physical fitness': df["Dance style"][i]}
                dict_list.append(data)
            elif ' and Muscle' in my_list[i]:
                physical_fitness.append(1)
                data = {'Increased physical fitness': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Enhanced physical fitness' in my_list[i]:
                physical_fitness.append(1)
                data = {'Increased physical fitness': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Weight loss' in my_list[i]:
                physical_fitness.append(1)
                data = {'Increased physical fitness': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Burning calories and losing' in my_list[i]:
                physical_fitness.append(1)
                data = {'Increased physical fitness': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Potential physical or mental health benefits associated with performing the dance. This describes' in my_list[i]:
                physical_fitness.append(1)
                data = {'Increased physical fitness': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Increased physical activity and exercise' in my_list[i]:
                physical_fitness.append(1)
                data = {'Increased physical fitness': df["Dance style"][i]}
                dict_list.append(data)
            else:
                physical_fitness.append(0)
            #print(physical_fitness)

        social_connection = []
        for i in range(len(my_list)):
            if " and Social connection" in my_list[i]:
                social_connection.append(1)
                data = {'Social connection': df["Dance style"][i]}
                dict_list.append(data)
            elif "Social connection" in my_list[i]:
                social_connection.append(1)
                data = {'Social connection': df["Dance style"][i]}
                dict_list.append(data)
            elif " Social connections" in my_list[i]:
                social_connection.append(1)
                data = {'Social connection': df["Dance style"][i]}
                dict_list.append(data)
            elif " and social connection" in my_list[i]:
                social_connection.append(1)
                data = {'Social connection': df["Dance style"][i]}
                dict_list.append(data)
            elif ' and Improved social skills' in my_list[i]:
                social_connection.append(1)
                data = {'Social connection': df["Dance style"][i]}
                dict_list.append(data)
            else:
                social_connection.append(0)
        #print(social_connection)

        mental_health = []
        for i in range(len(my_list)):
            if " and improves mental health" in my_list[i]:
                mental_health.append(1)
                data = {'Improves mental health': df["Dance style"][i]}
                dict_list.append(data)
            elif "Improves mental health" in my_list[i]:
                mental_health.append(1)
                data = {'Improves mental health': df["Dance style"][i]}
                dict_list.append(data)
            elif " Improves mental health" in my_list[i]:
                mental_health.append(1)
                data = {'Improves mental health': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Reduces stress and improves mental health' in my_list[i]:
                mental_health.append(1)
                data = {'Improves mental health': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Dancing can offer physical and mental health benefits' in my_list[i]:
                mental_health.append(1)
                data = {'Improves mental health': df["Dance style"][i]}
                dict_list.append(data)
            elif 'The Zapateado dance has the potential to offer numerous physical and mental health' in my_list[i]:
                mental_health.append(1)
                data = {'Improves mental health': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Potential physical or mental health benefits associated with performing the dance. This describes' in my_list[i]:
                mental_health.append(1)
                data = {'Improves mental health': df["Dance style"][i]}
                dict_list.append(data)
            elif 'potential_physical_mental_health_benefits' in my_list[i]:
                mental_health.append(1)
                data = {'Improves mental health': df["Dance style"][i]}
                dict_list.append(data)
            else:
                mental_health.append(0)
        #print(mental_health)

        strength = []
        for i in range(len(my_list)):
            if "Improved flexibility and strength" in my_list[i]:
                strength.append(1)
                data = {'Improved flexibility and strength': df["Dance style"][i]}
                dict_list.append(data)
            elif "Improved strength and flexibility" in my_list[i]:
                strength.append(1)
                data = {'Improved flexibility and strength': df["Dance style"][i]}
                dict_list.append(data)
            else:
                strength.append(0)
        #print(strength)

        community = []
        for i in range(len(my_list)):
            if " A sense of community and belonging" in my_list[i]:
                community.append(1)
                data = {'A sense of community and belonging': df["Dance style"][i]}
                dict_list.append(data)
            else:
                community.append(0)
        #print(community)

        boost = []
        for i in range(len(my_list)):
            if " Boost" in my_list[i]:
                boost.append(1)
                data = {'Boost': df["Dance style"][i]}
                dict_list.append(data)
            elif "Boost" in my_list[i]:
                boost.append(1)
                data = {'A sense of community and belonging': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Stress relief and improved mood' in my_list[i]:
                boost.append(1)
                data = {'A sense of community and belonging': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Boosted mood'in my_list[i]:
                boost.append(1)
                data = {'A sense of community and belonging': df["Dance style"][i]}
                dict_list.append(data)
            elif ' and Increased energy' in my_list[i]:
                boost.append(1)
                data = {'A sense of community and belonging': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Increased energy levels' in my_list[i]:
                boost.append(1)
                data = {'A sense of community and belonging': df["Dance style"][i]}
                dict_list.append(data)
            elif '55]: Increased happiness and well-being'in my_list[i]:
                boost.append(1)
                data = {'A sense of community and belonging': df["Dance style"][i]}
                dict_list.append(data)
            else:
                boost.append(0)
        #print(boost)

        cerebellum = []
        for i in range(len(my_list)):
            if " Improved balance" in my_list[i]:
                cerebellum.append(1)
                data = {'Improved cerebellum': df["Dance style"][i]}
                dict_list.append(data)
            elif "Improved balance" in my_list[i]:
                cerebellum.append(1)
                data = {'Improved cerebellum': df["Dance style"][i]}
                dict_list.append(data)
            elif ' and improved coordination' in my_list[i]:
                cerebellum.append(1)
                data = {'Improved cerebellum': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Improved coordination' in my_list[i]:
                cerebellum.append(1)
                data = {'Improved cerebellum': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Improved balance and coordination' in my_list[i]:
                cerebellum.append(1)
                data = {'Improved cerebellum': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Improved cognitive function' in my_list[i]:
                cerebellum.append(1)
                data = {'Improved cerebellum': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Improved posture' in my_list[i]:
                cerebellum.append(1)
                data = {'Improved cerebellum': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Improved balance and' in my_list[i]:
                cerebellum.append(1)
                data = {'Improved cerebellum': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Improved balance and coord' in my_list[i]:
                cerebellum.append(1)
                data = {'Improved cerebellum': df["Dance style"][i]}
                dict_list.append(data)
            elif ' Improved posture' in my_list[i]:
                cerebellum.append(1)
                data = {'Improved cerebellum': df["Dance style"][i]}
                dict_list.append(data)
            elif 'Improved posture' in my_list[i]:
                cerebellum.append(1)
                data = {'Improved cerebellum': df["Dance style"][i]}
                dict_list.append(data)
            else:
                cerebellum.append(0)
        #print(cerebellum)

        benefits = pd.DataFrame.from_dict(dict_list)
        st.dataframe(benefits)

    #Create a matrix for each health benefit
    health_benefits_matrix = pd.DataFrame([Improved_cardiovascular_health, Improved_flexibility, Stress_relief, confidence, physical_fitness,social_connection, mental_health, strength, community, boost, cerebellum])
    plt.figure(figsize=(10, 8))
    sns.heatmap(health_benefits_matrix, annot=True, cmap='coolwarm')
    plt.title('Heatmap for Health Benefits Matrix')
    #The most prevalent health beenfits in this dataset are the first 3: cardiovascular, flexibility, and stress relief


    #Filter data
    #filter = st.selectbox("Filter by Health Benefit", benefits["Improved cardiovascular health"].unique())
    #filtered = benefits[benefits["Improved cardiovascular health"]  == filter]
    #st.write(filtered)

    #filter = st.selectbox("Filter by Health Benefit", benefits["Improved flexibility"].unique())
    #filtered = benefits[benefits["Improved flexibility"]  == filter]
    #st.write(filtered)

if section == 'Conclusion':
    st.header("Conclusion")

    st.write('''
             
             Based on the analysis, I found that the tempo did not have a correlation to health benefits.

             Key Findings:
             - The variable "health benefits" requires further analysis using datasets that provide more comprehensive numerical data for deeper insights. The correlation analysis plot reveals a slight negative correlation, supporting this conclusion.
             - In the current calories dataset, the lack of detailed information and the limited number of dances did not yield enough meaningful insights. While the dance dataset provided specific details, the calories dataset overgeneralized.
             - Moving forward, I plan to identify a dataset that better aligns with the dance dataset, offers more relevant data, and incorporates more numerical variables. Additionally, I will treat the "health benefits" variable as a standalone dataset for focused analysis. - I will also seek datasets with numerical missing values to explore imputation techniques.
             ''')


#needs sidebards, tabs, dropdowns, tables, etc.
# must use advanced intercative visualization

# Object notation
#st.sidebar.[intro_page]
# "with" notation
#with st.sidebar:
    #st.[intro_page]




#can import data as csv file
#can make graphs
#multipage applications - that will create a sidebar
#can link to different page on the app -> st.link_button ("Profile", "/profile?id=1234"), but will open into a new window

#streamlit run /Users/kendallandrews/Downloads/streamtut.py
