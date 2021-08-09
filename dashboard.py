import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

def draw():
	df = pd.read_csv('creditcard.csv')
	st.title('Description of Dataset')
	st.write(df.describe())
	df['amount'] = df['Amount']
	df = df.drop(['Amount'], axis = 1)
	st.write('Checking for NaN values')
	st.write(df.isnull().sum())
	corr_mat = df.corr()
	st.title('Correlation Matrix')
	fig = go.Figure(go.Heatmap(z=corr_mat.values.tolist(), x=corr_mat.index.tolist(), y=corr_mat.columns.tolist(), colorscale='Blues'))
	st.write(fig)
	st.write(df.corrwith(df['Class']).plot.bar(figsize = (20,10)))
	true = df[df['Class']==0]
	false = df[df['Class']==1]
	st.title('Scatter Plot of True amounts')
	fig, ax = plt.subplots(figsize=(20,10))
	plt.scatter(true['Time'], true['amount'])
	st.write(fig)
	st.title('Scatter Plot of False amounts')
	fig, ax = plt.subplots(figsize=(20,10))
	plt.scatter(false['Time'], false['amount'])
	st.write(fig)
	st.title('Number of transactions which are normal/fraud')
	st.write(plt.bar(['Normal', 'Fraud'],[len(true), len(false)]))
	st.title('Description of Normal transactions')
	st.write(true.describe())
	st.title('Description of Fraudulent transactions')
	st.write(false.describe())
	X = df.drop('Class', axis=1)
	y = df['Class']
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
	X_train, y_train = SMOTE().fit_resample(X_train, y_train)
	st.title('Comparison of Fraud and Normal Transactions after SMOTE Handling')
	st.write(plt.bar(['Normal', 'Fraud'], [len(y_train==0), len(y_train==1)]))

    
def plotting_func(text_addr, s, mat):
    st.title(s)
    with open(text_addr, 'r') as f:
        st.write(f.readlines())
    st.write('Confusion Matrix')
    fig = ff.create_annotated_heatmap(z = mat, y = ['Predicted Normal','Predicted Fraud'], x=['Real Normal','Real Fraud'], colorscale = 'Blues')
    st.write(fig)

draw()
plotting_func('log_res.txt', 'Logistic Regression', np.array([[55376, 10], [1483, 93]]))
plotting_func('rfc.txt', 'Random Forest Classifier', np.array([[56848, 10], [11, 93]]))
plotting_func('sgd.txt', 'Stochastic Gradient Descent', np.array([[55132, 8], [1727, 95]]))
plotting_func('xgb.txt', 'XGB Classifier', np.array([[56836, 9], [23, 94]]))
plotting_func('svc.txt', 'C-Support Vector', np.array([[56709, 18], [150, 85]]))

