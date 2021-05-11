import streamlit as st 
import streamlit.components.v1 as stc 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Machine learning Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from models import run_model

#------------------------------------#
#util 
#funtion to load data
@st.cache()
def load_data(file):
	df = pd.read_csv(file)
	return df


#main
st.title('Churn for Bank Customers')
menu = ['EDA', 'Modelo ML', 'About']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'EDA':
	
	df = load_data("C:/Users/User/Documents/Cursos de ML/nicolas renotte/Machine Learning for Industry/Banking-Insurance/Churn/churn.csv")
	st.write(df.head())
	st.subheader('Exploratory Data Analysis')
	
	rad_but =  st.radio('Composición', ['filas', 'columnas'])
	if rad_but == 'filas':
		st.write(df.shape[0])
	else:
		st.write(df.shape[1])

	st.write('#### Typos de datos' )
	st.write(df.dtypes.value_counts())

	st.subheader('Visualización de los datos')
	st.write('#### Composición de la Clase' )
	clase = df['Exited'].value_counts()
	st.bar_chart(clase)




	object_columns = ['Geography', 'Gender', 'Exited']
	new_df =  df[object_columns]
	col1, col2 = st.beta_columns(2)
	with col1:
		st.write('Clasificación de clientes por Geografía')
		sns.countplot(data= new_df, x = 'Geography', hue = 'Exited')
		st.pyplot()

	with col2:
		st.write('Clasificación de clientes por Genero')
		sns.countplot(data= new_df, x = 'Gender', hue = 'Exited')
		st.pyplot()
	

	#Numerical Cols
	st.write('### Histograma')
	cols_view = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']	
	select_num = st.selectbox('Columnas Numericas', cols_view)
	df_hist = df[select_num]
	plt.figure(figsize=(20,60), facecolor='white')
	sns.histplot(df_hist, kde=True)
	st.pyplot()

elif choice == 'Modelo ML':
	run_model()

else:
	st.subheader('About')
	st.text('created by: luishernand11@gamil.com')