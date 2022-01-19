from flask import Flask
import matplotlib.pyplot as plt
from flask import render_template,request,url_for
import os
import numpy
import cv2

Car=	( 0, 0,255)	
Road=(255, 0, 0)	
Mark=	(255,255, 0)	
Building=	( 0,255, 0)	
Sidewalk=	(255, 0,255)	
Tree_Bush=	( 0,255,255)	
Pole	=(255, 0,153)
Sign	=(153, 0,255)	
Person=	( 0,153,255)	
Wall	=(153,255, 0)	
Sky	=(255,153,0)
Curb=	( 0,255,153)	
Grass_Dirt=	( 0,153,153)
Void	=( 0, 0, 0)	
Side_rail=	(153,153,153)
Object=	( 0, 0,153)
Bicycle_Motorbike=	(255,255,153)

###################
#Convert labeled images back to original RGB colored masks. 

def label_to_rgb(predicted_image):    
    
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))
    
    segmented_img[(predicted_image == 0)] = Car
    segmented_img[(predicted_image == 1)] = Road
    segmented_img[(predicted_image == 2)] = Mark
    segmented_img[(predicted_image == 3)] = Building
    segmented_img[(predicted_image == 4)] = Sidewalk
    segmented_img[(predicted_image == 5)] = Tree_Bush
    segmented_img[(predicted_image == 6)] = Pole
    segmented_img[(predicted_image == 7)] = Sign
    segmented_img[(predicted_image == 8)] = Person
    segmented_img[(predicted_image == 9)] = Wall
    segmented_img[(predicted_image == 10)] = Sky
    segmented_img[(predicted_image == 11)] = Curb
    segmented_img[(predicted_image == 12)] = Grass_Dirt
    segmented_img[(predicted_image == 13)] = Void
    segmented_img[(predicted_image == 14)] = Side_rail
    segmented_img[(predicted_image == 15)] = Object
    segmented_img[(predicted_image == 16)] = Bicycle_Motorbike
    
    segmented_img = segmented_img.astype(np.uint8)
    return(segmented_img)


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename



s=numpy.array([1,2,3])
app=Flask(__name__,template_folder='templates')
UPLOAD_FOLDER=r'D:\deploy\static'

# Model saved with Keras model.save()
MODEL_PATH = r'C:\Users\lenovo\Downloads\Transfer_Learning_Augemnted_final.h5'

# Load your trained model
model = load_model(MODEL_PATH,compile=False)
model.make_predict_function()

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model=model):
    img = cv2.imread(img_path,1) 
    W,H,_=img.shape
    img = cv2.resize(img, (256, 256))
 
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    return preds,W,H
#out,W,H=model_predict(r'D:\application\Deployment-Deep-Learning-Model-master\00_001200.png')



@app.route('/',methods=['GET','POST'])
def upload_predict():
    if request.method=='POST':
         image_file=request.files['image']
         if image_file:
             image_location=os.path.join(
                 UPLOAD_FOLDER,
                 "02.png"
             )
             image_file.save(image_location)
             pred,W,H=model_predict(image_location)
             result = label_to_rgb(np.argmax(pred, axis=3)[0])
             result = cv2.resize(result, (H, W))
            #  cv2.imwrite(r'D:\deploy\static\02.png', cv2.cvtColor(image_file, cv2.COLOR_RGB2BGR))
             cv2.imwrite(r'D:\deploy\static\01.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
             return render_template('index.html',prediction='Done',image_loc=image_file.filename)

    return render_template('index.html',prediction=0,image_loc=None)
@app.route('/home')
def home():
    return str(s[0])

if __name__=='__main__':
    app.run(debug=True)