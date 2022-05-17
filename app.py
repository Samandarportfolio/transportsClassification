from http.client import PRECONDITION_FAILED
from fastai.vision.all import *
import streamlit as st
import plotly.express as px
import pathlib


#title
st.title("Transportlarni klassifikatsiyalovchi model")

#file yuklaymiz
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'hiec', 'gif', 'webp'])
 
if file:
    #Rasmnni joylash
    st.image(file)

    #PILImage convert
    img = PILImage.create(file)

    #model
    model = load_learner('transportmodel.pkl')

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f'Bashorat : {pred}')
    st.info(f'Ehtimollik : {probs[pred_id]*100:.1f}%')

    #Plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)