import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier

from ml_app import run_ml_app


def main():
    st.title('당신이 닮은 포켓몬은? 과연???')


    # 사이드바 메뉴
    menu= ['Home','ML']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.write('이 앱은 CNN을 활용한 이미지 분류 서비스입니다.')
        st.write('당신과 가장 닮은 포켓몬은 누구일까요? (피카츄, 파이리, 꼬부기,이상해씨 중 1)')
        st.write('왼쪽의 사이드바에서 ML을 선택하세요.')
        st.image('data/pokemon.jpg')

    elif choice =='ML':
        run_ml_app()

if __name__ == '__main__':
    main()