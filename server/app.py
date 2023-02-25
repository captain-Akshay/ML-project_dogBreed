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

unique_breed = ['affenpinscher', 'Afghan Hound', 'African Hunting_dog', 'airedale',
                'American Staffordshire Terrier', 'appenzeller',
                'australian_terrier', 'basenji', 'basset', 'beagle',
                'bedlington_terrier', 'bernese Mountain_dog',
                'black and tan Coonhound', 'Blenheim Spaniel', 'Bloodhound',
                'bluetick', 'border_collie', 'border_terrier', 'borzoi',
                'boston_bull', 'bouvier_des_flandres', 'boxer',
                'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
                'cairn', 'cardigan', 'chesapeake Bay Retriever', 'chihuahua',
                'chow', 'clumber', 'cocker Spaniel', 'collie',
                'curly Coated_Retriever', 'dandie Dinmont', 'dhole', 'dingo',
                'doberman', 'english Foxhound', 'english Setter',
                'english Springer', 'entlebucher', 'eskimo Dog',
                'flat Coated Retriever', 'french Bulldog', 'german Shepherd',
                'german_short Haired pointer', 'giant Schnauzer',
                'golden Retriever', 'Gordon Setter', 'Great Dane',
                'great Pyrenees', 'greater Swiss Mountain Dog', 'groenendael',
                'Ibizan Hound', 'Irish Setter', 'Irish Terrier',
                'Irish Water Spaniel', 'Irish Wolfhound', 'italian Greyhound',
                'japanese Spaniel', 'keeshond', 'kelpie', 'kerry Blue Terrier',
                'komondor', 'kuvasz', 'Labrador Retriever', 'lakeland_terrier',
                'leonberg', 'lhasa', 'malamute', 'malinois', 'Maltese Dog',
                'Mexican Hairless', 'Miniature Pinscher', 'miniature Poodle',
                'Miniature Schnauzer', 'newfoundland', 'Norfolk Terrier',
                'Norwegian Elkhound', 'Norwich Terrier', 'Old English Sheepdog',
                'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
                'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
                'saint Bernard', 'saluki', 'samoyed', 'schipperke',
                'Scotch Terrier', 'Scottish Deerhound', 'Sealyham Terrier',
                'Shetland Sheepdog', 'Shih Tzu', 'Siberian Husky', 'Silky Terrier',
                'Soft Coated Wheaten Terrier', 'Staffordshire Bullterrier',
                'Standard Poodle', 'Standard Schnauzer', 'Sussex Spaniel',
                'Tibetan Mastiff', 'Tibetan Terrier', 'Toy Poodle', 'Toy Terrier',
                'vizsla', 'Walker Hound', 'weimaraner', 'Welsh Springer Spaniel',
                'west Highland White Terrier', 'Whippet',
                'Wire Haired fox Terrier', 'yorkshire Terrier']


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
