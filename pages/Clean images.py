import streamlit as st
import numpy as np
from PIL import Image
import json
import torch
import torch.nn as nn
import time
from torchvision import io
from torchvision import transforms as T
from torch.nn import functional as F
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd

# Класс модели 
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder 
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.SELU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=2),
            nn.BatchNorm2d(8),
            nn.SELU()
            )
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True, ceil_mode=True) #<<<<<< Bottleneck
        
        #decoder
        # Как работает Conv2dTranspose https://github.com/vdumoulin/conv_arithmetic

        self.unpool = nn.MaxUnpool2d(2, 2)
        
        self.conv1_t = nn.Sequential(
            nn.ConvTranspose2d(8, 32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.SELU()
            )
        self.conv2_t = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid()
            )     

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x, indicies = self.pool(x) # ⟸ bottleneck
        return x, indicies

    def decode(self, x, indicies):
        x = self.unpool(x, indicies)
        x = self.conv1_t(x)
        x = self.conv2_t(x)
        return x

    def forward(self, x):
        latent, indicies = self.encode(x)
        out = self.decode(latent, indicies)      
        return out
    
device = torch.device('cpu')

model = ConvAutoencoder()
model.load_state_dict(torch.load('weights/denoiserAE_weights_epoch_100.pt', map_location=device))
model.to(device)

# Метрики
n_epochs = 100
rmse = pd.read_csv('metrics.csv').drop('Unnamed: 0', axis=1)
data_size = pd.read_csv('data_sizes.csv')

# Функция предсказания
def get_prediction(image):

    transform = T.Compose([
        T.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)
    grayscale_image = torch.mean(image_tensor, dim=0, keepdim=True)
    
    model.to(device)
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        output = model(grayscale_image)
    
    end_time = time.time() 
    elapsed_time = end_time - start_time # Время работы модели 
    
    # Преобразование выходных данных обратно в изображение
    output_image = T.ToPILImage()(output.squeeze().cpu())

    return output_image, elapsed_time
    

st.write("## Denoiser ")
st.sidebar.write("Choose application above")
st.markdown(
        """
        This streamlit-app can clean your image 
    """
)
button_style = """
    <style>
    .center-align {
        display: flex;
        justify-content: center;
    
    </style>
"""
image_source = st.radio("Choose the option of uploading the image:", ("File", "URL"))

try:
    if image_source == "File":
        uploaded_files = st.file_uploader("Upload the image", type=["jpg", "png", "jpeg"], accept_multiple_files=True) 
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                st.image(image, caption="Original image", use_column_width=True)
                
                if st.button(f"Clean {uploaded_file.name}"):
                    cleaned_image, elapsed_time = get_prediction(image)
                    st.image(cleaned_image, caption="Cleaned image", use_column_width=True)
                    st.info(f"Время ответа модели: {elapsed_time:.4f} секунд")
            
            st.subheader("Model's training info")     
            st.info(f"Число эпох обучения: {n_epochs}")
            # выборка
            # fig, ax = plt.subplots()
            # ax.pie(data_size['Values'], labels=data_size['Categories'], autopct='%1.1f%%', startangle=90)
            # ax.axis('equal')
            # ax.set_title('Объем выборки')
            # st.pyplot(fig)

            fig, ax = plt.subplots()
            ax.bar(data_size['Categories'], data_size['Values'])
            # Настройки диаграммы
            ax.set_ylabel('Количество')
            ax.set_xlabel('Категории')
            ax.set_title('Объем выборки')
            st.pyplot(fig)

            # метрика
            fig, ax = plt.subplots()
            rmse.plot(kind='line', ax=ax)
            plt.title('Train and Valid RMSE Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            st.pyplot(fig)
            

    
    else:
        url = st.text_input("Enter the URL of image...")
        if url:
            response = requests.get(url)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.image(image, caption="Uploaded image", use_column_width=True)
                
                if st.button(f"Clean"):
                    cleaned_image, elapsed_time = get_prediction(image)
                    st.image(cleaned_image, caption="Cleaned image", use_column_width=True)
                    st.info(f"Время ответа модели: {elapsed_time:.4f} секунд")
            
                st.subheader("Model's training info")     
                st.info(f"Число эпох обучения: {n_epochs}")
                # выборка
                fig, ax = plt.subplots()
                ax.pie(data_size['Values'], labels=data_size['Categories'], autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                ax.set_title('Объем выборки')
                st.pyplot(fig) 
                # метрика
                fig, ax = plt.subplots()
                rmse.plot(kind='line', ax=ax)
                plt.title('Train and Valid RMSE Over Time')
                plt.xlabel('Epoch')
                plt.ylabel('RMSE')
                st.pyplot(fig) 
            else:
                st.error("Ошибка при получении изображения. Убедитесь, что введена правильная ссылка.")
except Exception as e:
    st.error(f"Произошла ошибка при обработке изображения {str(e)}")

st.markdown(button_style, unsafe_allow_html=True)  # Применяем стиль к кнопке
        