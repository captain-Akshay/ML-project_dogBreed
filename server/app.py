from __future__ import division, print_function
import os
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import load_model

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
MODEL_PATH = '../model/final_model.h5'

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"KerasLayer": hub.KerasLayer})

unique_breed = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
                'american_staffordshire_terrier', 'appenzeller',
                'australian_terrier', 'basenji', 'basset', 'beagle',
                'bedlington_terrier', 'bernese_mountain_dog',
                'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
                'bluetick', 'border_collie', 'border_terrier', 'borzoi',
                'boston_bull', 'bouvier_des_flandres', 'boxer',
                'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
                'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
                'chow', 'clumber', 'cocker_spaniel', 'collie',
                'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
                'doberman', 'english_foxhound', 'english_setter',
                'english_springer', 'entlebucher', 'eskimo_dog',
                'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
                'german_short-haired_pointer', 'giant_schnauzer',
                'golden_retriever', 'gordon_setter', 'great_dane',
                'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
                'ibizan_hound', 'irish_setter', 'irish_terrier',
                'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
                'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
                'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
                'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
                'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
                'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
                'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
                'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
                'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
                'saint_bernard', 'saluki', 'samoyed', 'schipperke',
                'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
                'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
                'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
                'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
                'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
                'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
                'west_highland_white_terrier', 'whippet',
                'wire-haired_fox_terrier', 'yorkshire_terrier']


def get_preds_label(predictions):
    return unique_breed[predictions.argmax()]


def process(imagepath):
    image = tf.io.read_file(imagepath)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[224, 224])
    return image


def create_batch(x, batch_size=32):
    # print("Test data is begin to be converted to batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    # print(1)
    data_batch = data.map(process).batch(batch_size)
    # print(2)
    return data_batch


def model_predict(img_path, model=model):
    img = create_batch(img_path)
    # print(img)
    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        pathh=[file_path]
        preds = model_predict(pathh, model)
        result = get_preds_label(preds)
        return result
    return None


if __name__ == '__main__':
    app.run()
