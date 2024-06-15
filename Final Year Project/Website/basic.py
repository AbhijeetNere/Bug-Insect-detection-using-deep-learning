
from flask import Flask , render_template,request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import mysql.connector 
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras

from keras.models import load_model
from keras.layers import DepthwiseConv2D
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
model_path = 'C:/Users/Abhay/Desktop/FY Project/Website/models/EfficientNetB4-INSECTS-0.76.h5'
model = load_model(model_path)
custom_objects = {'DepthwiseConv2D': DepthwiseConv2D}
model = load_model(model_path, custom_objects=custom_objects)


classes = ['Africanized Honey Bees (Killer Bees)',
'Aphids',
'Armyworms',
'Brown Marmorated Stink Bugs',
'Cabbage Loopers',
'Citrus Canker',
'Colorado Potato Beetles',
'Corn Borers',
'Corn Earworms',
'Fall Armyworms',
'Fruit Flies',
'Spider Mites',
'Thrips',
'Tomato Hornworms',
'Western Corn Rootworms'
]

db_config={
    'host': 'localhost',
    'user': 'root',
    'password': 'password',
    'database': 'newinsect'
}

app=Flask(__name__)

@app.route('/')
def home():
    return render_template("app.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('app.html', prediction_text="No file uploaded!")

    file = request.files['file']
    
    if file.filename == '':
        return render_template('app.html', prediction_text="No file selected!")
    

      # Generate a unique filename
    filename = f"temp_{np.random.randint(0, 100000)}.jpg"
    # Save the uploaded image to a temporary location
    fix='C:/Users/Abhay/Desktop/FY Project/Website/static/uploads/'
    img_path = os.path.join('static','uploads', filename)
    print("##### ther print is ; ",img_path)
    file.save(img_path)
    # Read the image file
    img_path = file.filename
    x=f'{fix}{filename}'
    #address=f"{fix}{filename}"
    img = plt.imread(x)
    
    img_size = (200, 200)
    img = cv2.resize(img, img_size)
    print ('the resized image has shape ', img.shape)
    img = np.expand_dims(img, axis=0)

    # Predict the image
    pred = model.predict(img)
    index = np.argmax(pred[0])
    klass = classes[index]
    probability = pred[0][index] * 100

    connection = None
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

                    # Query to fetch data based on the predicted class
        query = f"SELECT * FROM name WHERE name = '{klass}'"
        cursor.execute(query)
        print(query)
                    # Fetch all rows
        data = cursor.fetchall()

        return render_template('app.html', prediction_text=f'The image is predicted as being {klass} with a probability of {probability:.2f} %', file=f'uploads/{filename}', data=data)

    except mysql.connector.Error as error:
        print(f"Error fetching data from MySQL: {error}")
        return render_template('app.html',  prediction_text=f'The image is predicted as being {klass} with a probability of {probability:.2f} %',  file=f'uploads/{filename}')
                
    finally:
        if connection is not None:
            if connection.is_connected():
                cursor.close()
                connection.close()

                
    return render_template('app.html', prediction_text=f'The image is predicted as being {klass} with a probability of {probability:.2f} %', file=f'uploads/{filename}')


    
@app.route('/about')
def about():
    return render_template("about.html")
@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/team')
def team():
    return render_template("team.html")

if __name__=="__main__":
    app.run(debug=True)