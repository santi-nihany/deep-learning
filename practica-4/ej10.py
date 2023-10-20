from skimage.measure import regionprops, find_contours
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage import io
import pandas as pd
import numpy as np
import glob
import os
from tensorflow import keras
from sklearn import preprocessing, metrics, model_selection


SAMPLE_IMAGE_PATH = "./dataFiles/test/0a4d7cbc-2522-4e51-968a-1a86d3b7ee19_5L.png"
TEST_PATH = "./dataFiles/test/"
TRAIN_PATH = "./dataFiles/train/"


def get_image_props(img):

    # busca umbral global con método estadístico de Otsu
    umbral = threshold_otsu(img)

    # binariza la imagen
    img_bn = (img > umbral)*1

    # cierra pequeños huecos/cortes que pudiera tener la img de la mano
    img_bn = closing(img_bn, square(3))

    # remueve artefactos que pudiera tener la img en los bordes
    img_lista = clear_border(img_bn)

    # obtiene valores geométricos a partir de las regiones (objetos "aislados") en la img
    regiones = regionprops(img_lista)

    return (regiones[0], img_lista)


def get_image_props_from_path(path):
    img = io.imread(path, as_gray=True)
    return get_image_props(img)


def show_image(axs, img, cmap='gray'):
    # remueve ejes de los graficos
    axs.get_xaxis().set_ticks([])
    axs.get_yaxis().set_ticks([])

    axs.imshow(img, cmap=cmap)


def generate_dataset(path_from, path_to):
    columnas = ['filled_area', 'major_axis_length', 'minor_axis_length',
                'perimeter', 'eccentricity', 'solidity', 'extent', 'num_fingers']

    archivos = glob.glob(path_from + '*.png')

    data = np.empty((0, len(columnas)))

    for i, archivo in enumerate(archivos):
        print('\rTransformando %d imágenes: %6.2f%%\n' %
              (len(archivos), 100*(i+1)/len(archivos)), end='')

        cant_dedos = int(archivo[-6])
        img = io.imread(archivo)

        props, img_lista = get_image_props(img)

        area = props.filled_area
        ej_mayor = props.major_axis_length
        ej_menor = props.minor_axis_length
        perim = props.perimeter
        excentr = props.eccentricity
        solidez = props.solidity
        extension = props.extent
        razon_ej = ej_menor/ej_mayor
        data = np.append(data, np.array(
            [[area, ej_mayor, ej_menor, perim, excentr, solidez, extension, cant_dedos]]), axis=0)
    df = pd.DataFrame(data=data, columns=columnas)
    df.to_csv(path_to, index=False)
    print('Archivo', os.path.basename(path_to), ' generado correctamente ✅\n')


generate_dataset(TRAIN_PATH, './dataFiles/fingers/train/train.csv')
generate_dataset(TEST_PATH, './dataFiles/fingers/test/test.csv')


def train_MLP(path_train):
    datos = pd.read_csv(path_train)
    X = np.array(datos.iloc[:, :-1])
    Y = np.array(datos.iloc[:, -1])
    nomClases = datos.iloc[:, -1].value_counts()

    # Target codificado como one-hot
    encoder = preprocessing.LabelEncoder()
    Y_nro = encoder.fit_transform(Y)
    Y_bin = keras.utils.to_categorical(Y_nro)

    # %% --- CONJUNTOS DE ENTRENAMIENTO Y VALIDACION ---
    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(
        X, Y_bin, test_size=0.20)  # , random_state=42)

    normalizador = preprocessing.StandardScaler()
    X_train = normalizador.fit_transform(X_train)
    X_val = normalizador.transform(X_val)

    ENTRADAS = X_train.shape[1]
    SALIDAS = Y_train.shape[1]
    OCULTAS = 8
    EPOCAS = 100
    PACIENCIA = 10
    ACTIVACION = 'tanh'
    OPTIMIZADOR = 'adam'
    # %% CONSTRUCCION DEL MODELO
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(OCULTAS, input_shape=[
              ENTRADAS], activation=ACTIVACION))
    model.add(keras.layers.Dense(SALIDAS, activation='softmax'))
    model.summary()  # -- muestra la cantidad de parámetros de la red

    # -- se utilizará SGD (descenso de gradiente esticástico),
    # -- MSE (error cuadrático medio) y ACCURACY como medida de performance
    model.compile(optimizer=OPTIMIZADOR,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # %% ENTRENAMIENTO

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PACIENCIA)

    # entrena el modelo y guarda la historira del progreso
    history = model.fit(X_train, Y_train, epochs=EPOCAS,
                        validation_data=(X_val, Y_val), callbacks=[early_stop])


train_MLP('./dataFiles/fingers/train/train.csv')
