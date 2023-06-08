import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
State_en = pickle.load(open('State_en.pkl', 'rb'))
Wind_Direction_en = pickle.load(open('Wind_Direction_en.pkl', 'rb'))
Weather_Condition_en = pickle.load(open('Weather_Condition_en.pkl', 'rb'))
Sunrise_Sunset_en = pickle.load(open('Sunrise_Sunset_en.pkl', 'rb'))
Start_Time_Segment_en = pickle.load(open('Start_Time_Segment_en.pkl', 'rb'))
End_Time_Segment_en = pickle.load(open('End_Time_Segment_en.pkl', 'rb'))

with open('min_max_lists.pkl', 'rb') as f:
    # Load all lists from pickle file
    lists = []
    while True:
        try:
            obj = pickle.load(f)
            if isinstance(obj, list):
                lists.append(obj)
        except EOFError:
            break

min_list = lists[0]
max_list = lists[1]
del min_list[7]
del max_list[7]
scaler = MinMaxScaler()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    float_features = [x for x in request.form.values()]
    features = (np.array(float_features)).tolist()
    df = pd.DataFrame([features], columns=['Start_Lat', 'Start_Lng', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
                                           'Wind_Speed(mph)', 'Precipitation(in)', 'State_en', 'Wind_Direction_en',
                                           'Weather_Condition_en', 'Sunrise_Sunset_en', 'Start_Time', 'End_Time'])

    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['End_Time'] = pd.to_datetime(df['End_Time'])

    df['Start_Time_Hour'] = df['Start_Time'].dt.hour.map("{:2}".format).astype(int)
    df['End_Time_Hour'] = df['End_Time'].dt.hour.map("{:2}".format).astype(int)

    df['Start_Time_Segment'] = pd.Categorical(
        df['Start_Time_Hour'].apply(lambda x:
                                    '0H-6H' if x < 6 else
                                    '6H-12H' if x < 12 else
                                    '12H-18H' if x < 18 else
                                    '18H-24H'),
        categories=['0H-6H', '6H-12H', '12H-18H', '18H-24H'],
        ordered=True
    )

    df['End_Time_Segment'] = pd.Categorical(
        df['End_Time_Hour'].apply(lambda x:
                                  '0H-6H' if x < 6 else
                                  '6H-12H' if x < 12 else
                                  '12H-18H' if x < 18 else
                                  '18H-24H'),
        categories=['0H-6H', '6H-12H', '12H-18H', '18H-24H'],
        ordered=True
    )

    df['State_en'] = df['State_en'].map(State_en)
    df['Wind_Direction_en'] = df['Wind_Direction_en'].map(Wind_Direction_en)
    df['Weather_Condition_en'] = df['Weather_Condition_en'].map(Weather_Condition_en)
    df['Sunrise_Sunset_en'] = df['Sunrise_Sunset_en'].map(Sunrise_Sunset_en)
    df['Start_Time_Segment_en'] = df['Start_Time_Segment'].map(Start_Time_Segment_en)
    df['End_Time_Segment_en'] = df['End_Time_Segment'].map(End_Time_Segment_en)

    df = df.drop(columns=['Start_Time', 'End_Time', 'Start_Time_Hour', 'End_Time_Hour', 'Start_Time_Segment',
                          'End_Time_Segment'])
    pd.set_option("display.max_columns", None)

    new_row_df = pd.DataFrame([min_list, max_list], columns=['Start_Lat', 'Start_Lng', 'Humidity(%)', 'Pressure(in)',
                                                             'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
                                                             'State_en', 'Wind_Direction_en', 'Weather_Condition_en',
                                                             'Sunrise_Sunset_en', 'Start_Time_Segment_en',
                                                             'End_Time_Segment_en'])

    df = df.append(new_row_df, ignore_index=True)
    # print(df)

    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df = df.drop([1, 2])
    # print(df)

    prediction = model.predict(df)
    return render_template('index.html', prediction_text="The severity of the accident is {}".format(prediction))


if __name__ == '__main__':
    app.run(debug=True, port=9000)
