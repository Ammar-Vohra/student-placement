from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import sys

from src.exception.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["GET", "POST"])
def prediction():
    try:
        if request.method=="GET":
            return render_template("home.html")
        
        else:
            data = CustomData(
                gender = request.form.get('gender'),
                race_ethnicity = request.form.get('race_ethnicity'),
                parental_level_of_education = request.form.get('parental_level_of_education'),
                lunch = request.form.get('lunch'),
                test_preparation_course = request.form.get('test_preparation_course'),
                reading_score = request.form.get('reading_score'),
                writing_score = request.form.get("writing_score")

            )

            pred_df = data.get_data_as_data_frame()
            print(pred_df)

            predict_pipeline = PredictPipeline()
            result = predict_pipeline.predict(pred_df)
            formatted_result = f"{result[0]:.2f}"
            return render_template("home.html", result=formatted_result)

    except Exception as e:
        raise CustomException(e,sys)





if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0') 