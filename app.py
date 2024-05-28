import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import csv

supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
# model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()


def recommend_cosmetics(skin_type):
    if skin_type == "Brown_spott":
        return """Remember, early detection and swift action are crucial when time is short. Prioritize the most effective control measures based on available resources and time constraints.:
        1. Resistant Varieties: Planting Brown Spot-resistant rice varieties is the most effective method of control. These varieties have genetic resistance to the disease, reducing the need for other control measures.
        2. Weed Control: Implement effective weed control measures to reduce competition for nutrients and minimize the spread of fungal spores carried by weeds.
        3. Seed Treatment: Treat rice seeds with fungicides or biocontrol agents to reduce the risk of seedborne infection and early establishment of the Brown Spot pathogen..
        4. Resistant Varieties (if feasible): If time allows for planning ahead, consider planting Brown Spot-resistant rice varieties in future plantings to minimize disease risk.
        """

    elif skin_type == "Rice___Healthy":
        return """IT s an Healthy Leaf KEEPIT UP
        """

    elif skin_type == "Bacterial_leaf_blight":
        return """1. Resistant Varieties: Planting Bacterial leaf blight-resistant rice varieties is the most effective way to manage the disease. Choose varieties that have demonstrated resistance to Leaf Blast in your region.

                2.Crop Rotation: Rotate rice with non-host crops to disrupt the disease cycle and reduce the buildup of fungal spores in the soil.

                3.Proper Water Management: Maintain proper water management practices to avoid over-irrigation, which can create conditions conducive to Leaf Blast development. Ensure good drainage to prevent waterlogging.

                4.Fertilization: Apply balanced fertilizers to maintain optimal soil fertility levels. Avoid excessive nitrogen fertilization, as this can increase susceptibility to Leaf Blast.

                5.Weed Control: Implement effective weed control measures to reduce competition for nutrients and minimize the spread of fungal spores carried by weeds
        """

    elif skin_type == "Leaf_smut":
        return """1. Weed Control: Implement effective weed control measures to reduce competition for nutrients and minimize the spread of fungal spores carried by weeds.

                2. Sanitation: Remove and destroy infected crop residues after harvest to reduce the source of fungal inoculum in the field. This helps prevent the spread of Neck Blast to new rice crops.

                3.Fungicide Application: Apply fungicides preventively during the flowering stage or at the first sign of Neck Blast symptoms. Consult with agricultural experts for recommended fungicides and application timings.

                4.Biological Control: Explore the use of biofungicides containing beneficial microorganisms or organisms that suppress the growth of the Neck Blast pathogen.

                """

    else:
        return "Please enter a valid rice leaf type."



def prediction(image_path):

    import tensorflow as tf
    model = tf.keras.models.load_model(r'C:\Users\raopr\Downloads\Flask Deployed App\Flask Deployed App\plant_classifier_model.h5')
    import numpy as np
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings(action='once')
    from tensorflow.keras.preprocessing import image
    def prepare(img_path):
        img = image.load_img(img_path, target_size=(225,225))
        x = image.img_to_array(img)
        x = x/255
        return np.expand_dims(x, axis=0)

    img_path = image_path
    predictions = model.predict([prepare(img_path)])
    skin_types =['Bacterial_leaf_blight','Brown_spot', 'Rice___Healthy', 'Leaf_smut']
    predicted_skin_type = skin_types[np.argmax(predictions)]
    # print(f'Predicted Skin Type: {predicted_skin_type}')

    # # Generate skincare recommendations based on the predicted skin type
    # recommendations = recommend_cosmetics(predicted_skin_type)
    # print('Skincare Recommendations:')
    # print(recommendations)


    
    return predicted_skin_type

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit222', methods=['POST'])
def submit222():
    text = request.form['textfield']
    with open('data.csv', 'a', newline='') as csvfile:
        fieldnames = ['text']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'text': text})
    return 'Review submitted successfully!'

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        # title = disease_info['disease_name'][pred]
        # description =disease_info['description'][pred]
        # prevent = disease_info['Possible Steps'][pred]
        # image_url = disease_info['image_url'][pred]
        # supplement_name = supplement_info['supplement name'][pred]
        # supplement_image_url = supplement_info['supplement image'][pred]
        # supplement_buy_link = supplement_info['buy link'][pred]
        # return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
        #                        image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)
        
        # # Generate skincare recommendations based on the predicted skin type
        recommendations = recommend_cosmetics(pred)
        print('Skincare Recommendations:')
        print(recommendations)

        return render_template('submit.html' , pred = pred, recommendations=recommendations) 





@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']),disease_name= list(supplement_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
