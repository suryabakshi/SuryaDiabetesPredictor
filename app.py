import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Flask(__name__)
model = tf.keras.models.load_model('Diabetes.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/DiabetesPredictionBySurya',methods=['POST'])
def DiabetesPredictionBySurya():
	'''
	For rendering results on HTML GUI
	'''
	Pregnancies=float(request.form['Pregnancies_in'])
	Glucose = float(request.form['Glucose_in'])
	BloodPressure = float(request.form['BloodPressure_in'])
	SkinThickness = float(request.form['SkinThickness_in'])
	Insulin = float(request.form['Insulin_in'])
	BMI = float(request.form['BMI_in'])
	DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction_in'])
	Age=float(request.form['Age_in'])
	features_in = np.array([[Pregnancies, Glucose,BloodPressure,SkinThickness, Insulin,BMI,DiabetesPedigreeFunction,Age]])

	Diabetic_prediction = model.predict(features_in)
	if Diabetic_prediction > 0.5:
		return render_template('index.html', prediction_text = "The person is diabetic")
	else:
		return render_template('index.html', prediction_text= "The person is not diabetic")

if __name__ == "__main__":
    app.run(debug=True)
