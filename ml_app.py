import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from PIL import Image,ImageFilter,ImageEnhance
import h5py
import tensorflow.keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle 
import joblib
from sklearn.ensemble import RandomForestClassifier
import cv2
from keras.preprocessing import image




def run_ml_app():
    st.subheader('피카츄,꼬북이,파이리,이상해씨 중 자신과 가장 닮은 포켓몬은 누구일까요?')

    st.image('data/pokemon2.jpg')
    image_file = None
     

    image_file = st.file_uploader("Upload Image", type=["png","jpg",'jpeg'])
    
    
    if image_file is None :
        st.write("사진을 업로드 해주세요. 정면사진일수록 좋습니다.")#예측한다.

    # model = pickle.load(open("data/pima.pickle.dat", "rb"))
    
    #이미지를 resize한다.

    # img = load_image(image_file)
    # img_resize = img.resize((int(150), int(150)))

    else : 
        st.image(image_file)
        save_image = Image.open(image_file)
        resize_image = save_image.resize((150, 150))
        # resize_image = save_image, target_size=(150,150))

        x = image.img_to_array(resize_image) 
        x = np.expand_dims(x, axis=0)


        images = np.vstack([x])

        model = tensorflow.keras.models.load_model("data/my_model.h5")

        classes = model.predict(images, batch_size=10)

        if str(classes[0])== "[1. 0. 0. 0.]":
            st.write( "당신은 귀요미 꼬북이를 닮았습니다")
        elif str(classes[0])== "[0. 1. 0. 0.]":
            st.write("당신은 시크한 이상해씨를 닮았습니다")
        elif str(classes[0])== "[0. 0. 1. 0.]":
            st.write("당신은 늠름한 파이리를 닮았습니다")
        elif str(classes[0])== "[0. 0. 0. 1.]":
            st.write("당신은 발랄한 피카츄를 닮았습니다")
        else:
            st.write(" 인식이 어렵습니다. 다른 사진을 이용해주세요.")

    #  x = image.img_to_array(img)
    #  x = np.expand_dims(x, axis=0)
    # result = model.predict(image_file)
    # st.write(result)



    #new_data = np.array([0, 0.36  , 0.875 , 0.09547739, 0.48979592])
    
    # new_data = new_data.reshape(1,-1)
    
    # # sc_X = joblib.load('data/sc_X.pkl')

    # # new_data = sc_X.transform(new_data)

    # y_pred = model.predict(new_data)

    # #st.write(predicted_data[0][0])
    
    # # sc_y = joblib.load('data/sc_y.pkl')

    # # y_pred_original = sc_y.inverse_transform( y_pred)

    # if st.button("예측 결과 확인하기"):
        
    #     if y_pred == [0]:
    #         st.write("예측결과는 당뇨일 가능성이 적습니다.")            
    #     else :
    #         st.write("예측결과는 당뇨일 가능성이 높습니다. 의사와 상담하세요.")
        
    #     # st.write(  "에측 결과입니다. {:,.1f} 달러의 차를 살 수 있습니다".format(y_pred_original[0][0])  )