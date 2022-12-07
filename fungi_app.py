import tensorflow as tf
#!pip install joblib
import streamlit as st
#import joblib
#model = joblib.load('model.pkl') 
model = tf.keras.models.load_model('my_model.hdf5')


st.write("""
         # Prediccion de especies de hongos
         """
         )
st.write("Este es un clasificador de Especies de hongos")
file = st.file_uploader("Por favor subir una imagen", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

batch_size = 32
img_height = 184
img_width = 245

def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(img_width, img_height),    interpolation=cv2.INTER_CUBIC))
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Por favor subir una imagen")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    st.write(np.argmax(prediction))
    
    
    if np.argmax(prediction) == 0:
        st.write("Es una Amanita_albidostipes!")
    elif np.argmax(prediction) == 1:
        st.write("Es una Amanita_fritillaria!")
    elif np.argmax(prediction) == 2:
        st.write("Es una Amanita_griseofolia!")
    elif np.argmax(prediction) == 3:
        st.write("Es una Amanita_sinensis!")
    elif np.argmax(prediction) == 4:
        st.write("Es una Atractosporocybe_inornata!")
    elif np.argmax(prediction) == 5:
        st.write("Es una Boletus_erythropus!")
    elif np.argmax(prediction) == 6:
        st.write("Es una Clitocybe_nebularis!")
    elif np.argmax(prediction) == 7:
        st.write("Es una Ganoderma_lucidum!")
    elif np.argmax(prediction) == 8:
        st.write("Es una Gomphidius_roseus!")
    elif np.argmax(prediction) == 9:
        st.write("Es una Gomphus_cf._oritentalis!")
    elif np.argmax(prediction) == 10:
        st.write("Es una Gymnopus_dryophilus!")
    elif np.argmax(prediction) == 11:
        st.write("Es una Hydnum_repandum!")
    elif np.argmax(prediction) == 12:
        st.write("Es una Hymenopellis_furfuracea!")
    elif np.argmax(prediction) == 13:
        st.write("Es una Hypsizygus_marmoreus!")
    elif np.argmax(prediction) == 14:
        st.write("Es una Lactarius_vividus!")
    elif np.argmax(prediction) == 15:
        st.write("Es una Lactarius_volemus!")
    elif np.argmax(prediction) == 16:
        st.write("Es una Lactifluus_pilosus!")
    elif np.argmax(prediction) == 17:
        st.write("Es una Lepista_nuda!")
    elif np.argmax(prediction) == 18:
        st.write("Es una Panus_giganteus!")
    elif np.argmax(prediction) == 19:
        st.write("Es una Retiboletus_fuscus!")
    elif np.argmax(prediction) == 20:
        st.write("Es una Rhodocollybia_butyracea!")
    elif np.argmax(prediction) == 21:
        st.write("Es una Rugiboletus_extremiorientalis!")
    elif np.argmax(prediction) == 22:
        st.write("Es una Russula_adusta!")
    elif np.argmax(prediction) == 23:
        st.write("Es una Russula_anatina!")
    elif np.argmax(prediction) == 24:
        st.write("Es una Russula_compacta!")
    elif np.argmax(prediction) == 25:
        st.write("Es una Russula_foetens!")
    elif np.argmax(prediction) == 26:
        st.write("Es una Russula_nigricans!")
    elif np.argmax(prediction) == 27:
        st.write("Es una Russula_rosea!")
    elif np.argmax(prediction) == 28:
        st.write("Es una Russula_virescens!")
    elif np.argmax(prediction) == 29:
        st.write("Es una Russula_viridirubrolimbata!")
    elif np.argmax(prediction) == 30:
        st.write("Es una Scleroderma_citrinum!")
    elif np.argmax(prediction) == 31:
        st.write("Es una Stropharia_rugosoannulata!")
    elif np.argmax(prediction) == 32:
        st.write("Es una Suillus_bovinus!")
    elif np.argmax(prediction) == 33:
        st.write("Es una Suillus_cavipes!")
    elif np.argmax(prediction) == 34:
        st.write("Es una Termitomyces_albuminosus!")
    elif np.argmax(prediction) == 35:
        st.write("Es una Tricholoma_bakamatsutake!")
    elif np.argmax(prediction) == 36:
        st.write("Es una Tricholoma_terreum!")
    elif np.argmax(prediction) == 37:
        st.write("Es una termitomyces_micro!")
    #else:
    #    st.write("Es una a scissor!")
    
    st.text("Probability (0: Amanita_albidostipes, 1: Amanita_fritillaria, 2: Amanita_griseofolia, 3: Amanita_sinensis, 4: Atractosporocybe_inornata, 5 :Boletus_erythropus, 6 :Clitocybe_nebularis, 7 :Ganoderma_lucidum, 8 :Gomphidius_roseus, 9 :Gomphus_cf._oritentalis, 10 :Gymnopus_dryophilus, 11 :Hydnum_repandum, 12 :Hymenopellis_furfuracea, 13 :Hypsizygus_marmoreus, 14 :Lactarius_vividus, 15 :Lactarius_volemus, 16 :Lactifluus_pilosus, 17 :Lepista_nuda, 18 :Panus_giganteus, 19 :Retiboletus_fuscus, 20 :Rhodocollybia_butyracea, 21 :Rugiboletus_extremiorientalis, 22 :Russula_adusta, 23 :Russula_anatina, 24 :Russula_compacta, 25 :Russula_foetens, 26 :Russula_nigricans, 27 :Russula_rosea, 28 :Russula_virescens, 29 :Russula_viridirubrolimbata, 30 :Scleroderma_citrinum, 31 :Stropharia_rugosoannulata, 32: Suillus_bovinus, 33 :Suillus_cavipes, 34 :Termitomyces_albuminosus, 35 :Tricholoma_bakamatsutake, 36 :Tricholoma_terreum, 37 :termitomyces_micro)")
    st.write(prediction)


