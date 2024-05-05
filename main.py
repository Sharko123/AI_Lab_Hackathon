from flask import Flask, render_template, jsonify, request
import numpy as np
import joblib
from xgboost import XGBRegressor
import pandas as pd

app = Flask(__name__)

model = joblib.load("xgb_model.pkl")


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
  cuisine = (request.form.get("cuisine"))
  category = (request.form.get("category"))
  city_enc_4 = (request.form.get("city_enc_4"))
  discount_yn = (request.form.get("discount_yn"))
  feature = (request.form.get("homepage_features"))
  email = (request.form.get("emailer_promotion"))
  checkout = (request.form.get("checkout_price"))
  base = (request.form.get("base_price"))
  discAmt = (request.form.get("discount_amount"))
  discPerc = (request.form.get("discount_percent"))

  data = [
      checkout, base, email, feature, discAmt, discPerc, discount_yn, category,
      cuisine, city_enc_4
  ]
  data_2d = [data]
  df = pd.DataFrame(data_2d,
                    columns=[
                        'checkout_price', 'base_price',
                        'emailer_for_promotion', 'homepage_featured',
                        'discount amount', 'discount percent', 'discount y/n',
                        'category', 'cuisine', 'city_enc_4'
                    ])
  df['city_enc_4'] = df['city_enc_4'].astype('object')
  df['cuisine'] = df['cuisine'].astype('object')
  df['category'] = df['category'].astype('object')
  df['discount y/n'] = df['discount y/n'].astype('float')
  df['homepage_featured'] = df['homepage_featured'].astype('float')
  df['emailer_for_promotion'] = df['emailer_for_promotion'].astype('float')
  df['checkout_price'] = df['checkout_price'].astype('float')
  df['base_price'] = df['base_price'].astype('float')
  df['discount amount'] = df['discount amount'].astype('float')
  df['discount percent'] = df['discount percent'].astype('float')
  missing_columns = [
      'category_Biryani', 'category_Desert', 'category_Extras',
      'category_Other Snacks', 'category_Pasta', 'category_Pizza',
      'category_Rice Bowl', 'category_Salad', 'category_Sandwich',
      'category_Seafood', 'category_Soup', 'category_Starters',
      'cuisine_Indian', 'cuisine_Italian', 'cuisine_Thai', 'city_enc_4_CH2',
      'city_enc_4_CH3', 'city_enc_4_CH4'
  ]
  obj = df[['category', 'cuisine', 'city_enc_4']]
  encode1 = pd.get_dummies(obj, drop_first=True)
  for column in missing_columns:
    if column not in encode1.columns:
      encode1[column] = False
  num = df.drop(columns=["city_enc_4", "category", "cuisine"], axis=1)
  df = pd.concat([num, encode1], axis=1)

  print(df)

  html_data = model.predict(df)
  html_data = int(html_data)
  print("ANSWER: ", html_data)

  return render_template("predict.html", html_data=html_data)


@app.route('/drive', methods=["POST", "GET"])
def drive():
  city = request.form.get("city")
  drives = search_food_drives(city)
  return render_template("drive.html", city=city, drives=drives)


if __name__ == '__main__':
  app.debug = False
  app.run()

import requests


def search_food_drives(city):
  api_key = 'AIzaSyDCeQNu6b5FaJZF8nyPc7mCLREVVxjG2NI'
  url = f'https://maps.googleapis.com/maps/api/place/textsearch/json'
  params = {'query': f'food drive near {city}', 'key': api_key}

  response = requests.get(url, params=params)
  data = response.json()
  final = []
  print(data)
  if 'results' in data:
    for result in data['results']:
      name = result['name']
      address = result['formatted_address']
      final.append(f'Name: {name} \n Address: {address}')
      print("FINAL: ", final)
    return final
  else:
    print('No food drives found.')
    final = "No food drives found."
    return final
