import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(
            "Physical GPUs: ", len(gpus),
            "Logical GPUs: ", len(logical_gpus)
        )
    except RuntimeError as e:
        print(e)


def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = resnet50.preprocess_input(img_batch)
    model = resnet50.ResNet50()
    prediction = model.predict(img_preprocessed)
    print(resnet50.decode_predictions(prediction, top=3)[0])


IMG_PATH = 'sample_images/cat.jpg'
predict(IMG_PATH)
