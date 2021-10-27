import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

data = pd.read_csv("https://drive.google.com/file/d/1kqJ54-Q8NLWbPnCeiAV67TXZn9B7lkv1/view?usp=sharing", decimal=',')
st.set_option('deprecation.showPyplotGlobalUse', False)

# image = Image.open('https://drive.google.com/file/d/1m9ledmPWCpKOYoDYDXerkekz7haJS302/view?usp=sharing')
# st.image(image)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file, sep=",")
  st.write(df)

st.title("Countplot du Nutriscore")

fig6 = px.histogram(df, x='nutriscore_grade', barmode='group')
st.plotly_chart(fig6)

st.title("Boxplot du Nutriscore")

fig7 = px.box(df, x='nutriscore_grade', y="fat_100g", notched=True)
st.plotly_chart(fig7)

st.title("Le Nutriscore en fonction des différents éléments")

fig1 = px.histogram(df, x="saturated-fat_100g", color="nutriscore_grade").update_xaxes(categoryorder="category ascending")
st.plotly_chart(fig1)

fig2 = px.histogram(df, x="energy_100g", color="nutriscore_grade").update_xaxes(categoryorder="category ascending")
st.plotly_chart(fig2)

fig3 = px.histogram(df, x="fat_100g", color="nutriscore_grade").update_xaxes(categoryorder="category ascending")
st.plotly_chart(fig3)

fig4 = px.histogram(df, x="sugars_100g", color="nutriscore_grade").update_xaxes(categoryorder="category ascending")
st.plotly_chart(fig4)

fig5 = px.histogram(df, x="salt_100g", color="nutriscore_grade").update_xaxes(categoryorder="category ascending")
st.plotly_chart(fig5)


st.title("TROUVE TON NUTRISCORE")

df_dep = pd.read_csv('https://drive.google.com/file/d/1kqJ54-Q8NLWbPnCeiAV67TXZn9B7lkv1/view?usp=sharing', decimal=',')
df_app = pd.DataFrame(index=['0'], columns=['energy_100g','energy-kcal_100g',
                                            'fat_100g','sugars_100g','saturated-fat_100g','salt_100g'])

energie = st.number_input('entrer le nombre de energie/100g: ')
energie_kcal = st.number_input('entrer le nombre de energie_kcal/100g: ')
fat = st.number_input('entrer le nombre de matières grasses: ')
sat_fat = st.number_input('entrer le nombre de graisse saturé: ')
sugar = st.number_input('entrer le nombre de sucre: ')
salt = st.number_input('entrer le nombre de sel: ')


df_app.at['0', 'energy_100g'] = energie
df_app.at['0', 'energy-kcal_100g'] = energie_kcal
df_app.at['0', 'fat_100g'] = fat
df_app.at['0', 'sugars_100g'] = sugar
df_app.at['0', 'saturated-fat_100g'] = sat_fat
df_app.at['0', 'salt_100g'] = salt

y_train =df_dep['nutriscore_grade']
X_train = df_dep.drop(['nutriscore_grade'], axis=1)
X_test = df_app

model = pickle.load(open("https://drive.google.com/file/d/1Y09vuHQIEutJcKIBzTF6Sg5UZLpq9hPZ/view?usp=sharing","rb"))

model = RandomForestClassifier(n_estimators= 100)
model.fit(X_train, y_train)

def score(x):
    if x ==5:
        return 'E'
    elif x ==4:
        return 'D'
    elif x == 3:
        return 'C'
    elif x == 2:
        return 'B'
    else:
        return 'A'
        
st.write('le nutriscore de votre produit est: ',score(y_pred))



# streamlit run C:/Users/Utilisateur/Desktop/Alyson/Flasks/Openfood/exemple.py




# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#   df = pd.read_csv(uploaded_file, sep=",")
#   st.write(df)
