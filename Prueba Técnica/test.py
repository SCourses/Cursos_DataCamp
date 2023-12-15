#libreriaa necesarias
#nltk
import nltk
nltk.download('stopwords')
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import string
import pickle
from tensorflow import keras
import re
import joblib
import os
cwd = os.getcwd()


"""******************** Texto Ejemplo ******************************"""
texto_ejemplo_poema="En la penumbra de la noche sosegada, donde las estrellas sus secretos revelan, mi mente vuela en alas de la nada, donde los sueños como luciérnagas destellan.Bajo el manto plateado de la luna, se tejen historias en el silencio, donde el corazón, como una fortuna, se llena de anhelos en este trance eterno.Susurros de hojas danzan con el viento, mientras la noche abraza el suspenso, y en el lienzo del cielo, un firmamento, pintado con los sueños que llevo dentro.Allí, donde los recuerdos se entrelazan, y las sombras danzan con la melodía, se despiertan los sueños que abrazan, la esperanza en esta mágica poesía.Que las estrellas guarden mis secretos, y la luna cuente mis anhelos, en este rincón de sueños discretos, donde la noche se convierte en cielo."





with open('./Models/modelo_Clasificacion.pkl', 'rb') as f:
    Clasificacion = pickle.load(f)

Tipo_Texto = keras.models.load_model('./Models/Tipo_texto.keras')
scaler=joblib.load('./Models/scaler_model.joblib')
encoder=joblib.load('./Models/encoder_model.joblib')
tokenize=joblib.load('./Models/tokenizer_model.joblib')
stop_words = set(stopwords.words('spanish'))

def extraccion_caracteristicas(texto):
    def total_palabras(texto):
        total=len(word_tokenize(texto))
        return total

    def numero_stop_words_por_texto(texto,):
        total=total_palabras(texto)
        texto=word_tokenize(texto)
        stopwords_x = [w for w in texto if w in stop_words]
        num_stopw=len(stopwords_x)
        num_stopw_per_total= num_stopw / total if total else 0
        return num_stopw_per_total

    def numero_palabras_unicas_por_texto(texto):
        total=total_palabras(texto)
        texto=word_tokenize(texto)
        unicos = len([*set(texto)])
        unicos_per_text=unicos / total if total else 0
        return unicos_per_text

    def palabras_mayusculas_por_texto(texto):
        total=total_palabras(texto)
        palabras_con_mayusculas = len([palabra for palabra in texto.split() if any(letra.isupper() for letra in palabra)])
        palabras_con_mayusculas_per_text=palabras_con_mayusculas / total if total else 0
        return palabras_con_mayusculas_per_text

    def signos_puntuacion_por_texto(texto):
        total=total_palabras(texto)
        cantidad_signos = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
        num=cantidad_signos(texto, string.punctuation)
        return num / total if total else 0

    def palabras_repetidas_por_texto(texto):
        total=total_palabras(texto)
        conteo_palabra = {}
        words = word_tokenize(texto)
        for palabra in words:
            # Remove punctuation if necessary, convert to lowercase, etc.
            palabra = palabra.strip().lower()  # Adjust as needed
            conteo_palabra[palabra] = conteo_palabra.get(palabra, 0) + 1
        num=np.sum(np.array(list(conteo_palabra.values()))-1)
        return num / total if total else 0
    

    return pd.Series([numero_stop_words_por_texto(texto),numero_palabras_unicas_por_texto(texto),signos_puntuacion_por_texto(texto),palabras_mayusculas_por_texto(texto),palabras_repetidas_por_texto(texto)])

# Se corrigen algunos caracteres existentes en los textos, como lo son errores en lectura de tíldes y simbolos caracteristicos usados en tweets
def prerpocesamiento(texto):
    diccionario_tildes= {
        "Ã¡": "á",
        "Ã©": "é",
        "Ã": "í",
        "Ã³": "ó",
        "Ãº": "ú",
        "Ã±": "ñ",
    }
    for mal_codificado, bien_codificado in diccionario_tildes.items():
        texto = texto.replace(mal_codificado, bien_codificado)

    #preprocesamiento para los tweets
    # remover el símbolo de retweet "rt"
    texto = re.sub(r'^RT[\s]+', '', texto)
    # remover los liks
    texto = re.sub(r'https?://[^\s\n\r]+', '', texto)

    # remover hashtags
    # solo se remueven los # de las palabras
    texto = re.sub(r'#', '', texto)

    return texto

REEMPLAZAR_ESPACIO = re.compile('[/(){}\[\]\|@,;]')
SIMBOLOS_INCORRECTOS = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('spanish'))

def prerpocesamiento_tipo_text(texto):
    """
        text:  string
        
        return: strin modificado
    """
    #diccionario en donde se reemplazan algunos caracteres mal procesados
    diccionario_tildes= {
        "Ã¡": "á",
        "Ã©": "é",
        "Ã": "í",
        "Ã³": "ó",
        "Ãº": "ú",
        "Ã±": "ñ",
    }
    for mal_codificado, bien_codificado in diccionario_tildes.items():
        texto = texto.replace(mal_codificado, bien_codificado)

    texto = texto.lower() # se convierte todo el texto a su versión en munuscula
    texto=texto.replace('\xad', '')
    texto = REEMPLAZAR_ESPACIO.sub(' ', texto) # replace REPLACE_BY_SPACE_RE symbols by space in text
    texto = SIMBOLOS_INCORRECTOS.sub('', texto) # delete symbols which are in BAD_SYMBOLS_RE from text
    texto = ' '.join(word for word in texto.split() if word not in STOPWORDS) # delete stopwors from text
    return texto



def prueba_modelo_calss(text):
    data=pd.DataFrame(index=[1])
    data[['stop_words/texto','numero_palabras_unicas/texto','signos_puntuacion/texto','palabras_mayusculas/texto','palabras_repetidas/texto']]=extraccion_caracteristicas(text)
    datos=scaler.transform(data)
    pred=Clasificacion.predict(datos)
    if pred[0]==0:
        clase="Humano"
    else:
        clase="LLM"
    
    return clase

def prueba_modelo_tipo(texto,tokenizer):
    texto_limpio=pd.Series([prerpocesamiento_tipo_text(texto)])
    tokenizado=tokenizer.texts_to_matrix(texto_limpio)
    prediccion=Tipo_Texto.predict(tokenizado)
    predict_tipo=encoder.inverse_transform([prediccion[0].argmax()])
    return predict_tipo[0]



print(prueba_modelo_tipo(texto_ejemplo_poema,tokenize))
print(prueba_modelo_calss(texto_ejemplo_poema))
