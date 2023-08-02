from flask.helpers import send_file
import pandas as pd
from flask import Flask,render_template,jsonify,request
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.pipelines.training_pipeline import DataTransformation

application=Flask(__name__)

app=application

@app.route('/')
def homepage():
    return ('Welcome To My Application')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        data=request.form.get('file')

        data=pd.DataFrame(data)
        predict_pipeline=PredictionPipeline(request)
        prediction_file_detail = predict_pipeline.run_pipeline()

        return send_file(prediction_file_detail.prediction_file_path,
                            download_name= prediction_file_detail.prediction_file_name,
                            as_attachment= True)
    
    else:
        return render_template('upload_file.html')

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)