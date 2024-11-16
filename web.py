import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot  as plt
import pickle
import seaborn as sns

# page title
st.title("Predictions App(in development)")

# get the data
st.write("Upload Dataset in csv format")
uploaded_file=st.file_uploader("Upload Data: ",type="csv" )

# Display the data
df=pd.read_csv(uploaded_file)
st.write("Uploaded Data: ", df.head())
st.write(f"{df.shape[0]} rows and {df.shape[1]} columns.")

# Clean
df.dropna(inplace=True)

# Target Variable
st.sidebar.header(" Select Target Variable")
col_names = list(df.columns)
Target_var=st.sidebar.selectbox("Target Variable", col_names)
y = df[Target_var] 
df=df.drop(columns=[Target_var]) # so it does not show to feature selections

# Features
num_col_names=df.select_dtypes(exclude="object").columns.to_list()
cat_col_names=df.select_dtypes(include="object").columns.to_list()
# df=pd.get_dummies(df[cat_col_names], drop_first=True) # some columns can be nominal causing lots of dummies. 

x_features=list(df[num_col_names]) # selecting only numerical features only for simple modeling

st.sidebar.header("Choose From Numerical Features: ")
selected_features=st.sidebar.multiselect("X features ", x_features)

# Build Model
X=df[selected_features]
st.write(X.head())

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.3, random_state=0)

# Model Selection
st.sidebar.header("Select Appropriate Model For Classification: ")
model_name = st.sidebar.selectbox("models", ("Logistic_Regression", "Decission_Tree", "Random_Forest"))


if model_name == "Logistic_Regression":
    model = LogisticRegression()
    model.fit(X_train, y_train)
    st.write("Train Accuracy:", model.score(X_train, y_train))
    st.write("Test Accuracy:", model.score(X_test, y_test))
    with open("model.pickle", "wb") as f:
        pickle.dump(model, f)

elif model_name == "Decission_Tree":
    st.sidebar.write("Decission Tree Parameters: ")
    max_depth = st.sidebar.slider("Max Depth: ", 2, 20)
    spliting_criterion=st.sidebar.selectbox("Criterion: ",("gini", "entropy"))
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=spliting_criterion)
    model.fit(X_train, y_train)
    st.write("Train Accuracy:", model.score(X_train, y_train))
    st.write("Test Accuracy:", model.score(X_test, y_test))
    with open("model.pickle", "wb") as f:
        pickle.dump(model, f)

elif model_name == "Random_Forest":
    st.write("Random Forest is not available right now.")
   


