import numpy as np
from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image

import streamlit as st
import urllib.request
import os

import os.path
from os import path
import shutil


with st.sidebar:

    st.header("🖥️เกี่ยวกับโปรเจคนี้")
    st.write("โปรเจคนี้อ้างอิงมากจากโครงงาน เว็บแอปพลิเคชั่นการคัดแยกโรคมะเร็งการเรียนรู้ถ่ายโอน Vision Transformer จัดทำโดย 1)นายภัคพล อาจบุราย 2)นายสันติภาพ ทองจันทร์ 3)นายณัฐวีร์ แนวกำพล อาจารย์ที่ปรึกษา นายเอกชัย วัฒนไชย ",unsafe_allow_html=True)
    
    st.header("🌐แหล่งอ้างอิง")
    st.write("ยังไม่เสร็จ")

st.header('Lung Cancer classification with Vision Transformer: จำแนกมะเร็งปอด')


with open("assets/webfonts/font.txt") as f:
    st.markdown(f.read(),unsafe_allow_html=True)
with open("assets/css/style.css") as f:
    st.markdown(f"<style> {f.read()} </style>",unsafe_allow_html=True)
hide_table_index = """
            <style>         
            thead {display:none}  
            tbody th {display:none}
            .blank {display:none}
            </style>
            """ 
st.markdown(hide_table_index, unsafe_allow_html=True)

#checkfile model
if path.exists("model/optimizer.pt") == True:
    print("optimizer installed")
else :
    #download model
    print("optimizer not installed")
    optimizer = "https://huggingface.co/alicelouis/VisLungTransformer/resolve/main/checkpoint-1644/optimizer.pt"
    urllib.request.urlretrieve(optimizer,"optimizer.pt")
    print("load optimizer")

    listModel = ['optimizer.pt']

    for i in range(len(listModel)):
        print(listModel[i])
        new_path = 'model/' + listModel[i]
        shutil.move(listModel[i] , new_path)
    
if path.exists("model/pytorch_model.bin") == True:
    print("pytorch_model installed")
else :
    #download model
    print("pytorch_model not installed")
    pytorch_model = "https://huggingface.co/alicelouis/VisLungTransformer/resolve/main/checkpoint-1644/pytorch_model.bin"
    urllib.request.urlretrieve(pytorch_model,"pytorch_model.bin")
    print("load pytorch_model")

    listModel = ['pytorch_model.bin']
    for i in range(len(listModel)):
        print(listModel[i])
        new_path = 'model/' + listModel[i]
        shutil.move(listModel[i] , new_path)
    

#model path
model_name_or_path = "model"
#labels
labels = ["adenocarcinoma",
 "large.cell.carcinoma",
 "normal",
 "squamous.cell.carcinoma"]
#load model
model = BeitForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)
#load fretureExtrator
feature_extractor = BeitFeatureExtractor.from_pretrained(model_name_or_path)


uploaded_file = st.file_uploader("อัปโหลดไฟล์ภาพ")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_out = img
    img_out = np.array(img_out)
    # โหลดโมเดลที่เซฟ
    image = img.resize((224,224))
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    # model predicts one of the 4 classes
    predicted_class_idx = logits.argmax(-1).item()
    className = labels[predicted_class_idx]

    st.success("Predicted class is : " + className , icon="✅")
    st.image(img_out)
