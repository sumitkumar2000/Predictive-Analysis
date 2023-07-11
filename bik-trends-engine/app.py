import os
from flask import Flask, jsonify, request
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.tools as tls
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import joblib
import plotly.io as pio
from google.cloud import storage
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)

data = pd.read_csv('time_series_data.csv', index_col=0)

@app.route("/")
def index():
    return "Hello from bik_trends_engine"


def best_arima_model(data):
    p = q = range(0, 10)
    pq = list(product(p, q))
    print(pq)
    models = {}
    for i in pq:
        try:
            model = ARIMA(data, order=(i[0], 0, i[1]))
            results = model.fit()
            print('results',results)
            models[i] = results.aic
        except ValueError as ve:
            print(str(ve))
            continue
    best_model = min(models, key=models.get)
    return best_model

@app.route('/predict', methods=['POST'])
def predict():
    data = pd.read_csv('time_series_data.csv')
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    # Get input data
    try:
        json_data = request.get_json(force=True)
        input_data = pd.DataFrame.from_dict(json_data)
        input_data['date'] = pd.to_datetime(input_data['date'])
        input_data.set_index('date', inplace=True)
        input_data.columns = ['values']
        input_data = input_data['values']
    except:
        return jsonify({'error':'Input format not matched'}), 400

    # Train the model
    try:
        order = best_arima_model(data)
        model = ARIMA(data, order=(order[0],0,order[1]))
        model_fit = model.fit()
    except ValueError as ve:
        return jsonify({'error here': str(ve)}), 400

    # Make predictions
    try:
        predictions = model_fit.predict(start=len(data), end=len(data)+len(input_data)-1)
        mse = mean_squared_error(input_data, predictions)
    except:
        return jsonify({'error': 'Failed to make predictions.'}), 500

    # Format predictions as JSON
    result = {'predictions': predictions.tolist(), 'mse': mse}

    # Return response as JSON
    return jsonify(result)

from flask import Flask, jsonify, request
import pandas as pd
import statsmodels.api as sm

app = Flask(__name__)

@app.route('/forecast', methods=['POST'])
def forecast_sales():
    # Get the input data
    data = request.get_json()

    # Convert the input data into a pandas DataFrame
    df = pd.DataFrame(data['sales'])

    # Preprocess the data
    df['week'] = pd.to_datetime(df['week'])
    df.set_index('week', inplace=True)

    # Split data into training and validation sets
    train_data = df.iloc[:-13]
    validation_data = df.iloc[-13:]

    # Define parameter combinations for grid search
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    P_values = [0, 1, 2]
    D_values = [0, 1]
    Q_values = [0, 1, 2]
    parameter_combinations = list(product(p_values, d_values, q_values, P_values, D_values, Q_values))

    best_mse = float('inf')
    best_params = None

    # Iterate over parameter combinations
    for params in parameter_combinations:
        p, d, q, P, D, Q = params


        # Fit SARIMA model
        model = sm.tsa.SARIMAX(train_data['sales'], order=(p, d, q), seasonal_order=(P, D, Q, 12),initialization='approximate_diffuse')
        result = model.fit()

        # Forecast using the validation set
        forecast = result.get_forecast(steps=len(validation_data))
        forecasted_sales = forecast.predicted_mean

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(validation_data['sales'], forecasted_sales)

        # Update the best parameters if the MSE improves
        if mse < best_mse:
            best_mse = mse
            best_params = params

    # Fit the final model with the best parameters
    p, d, q, P, D, Q = best_params
    model = sm.tsa.SARIMAX(df['sales'], order=(p, d, q), seasonal_order=(P, D, Q, 12))
    final_result = model.fit()

    # Forecast for Q3
    forecast_steps = 13
    forecast = final_result.get_forecast(steps=forecast_steps)

    # Retrieve the forecasted values
    forecasted_sales = forecast.predicted_mean

    # Prepare the forecast data for response
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_steps, freq='W-SAT').strftime('%Y-%m-%d')
    forecast_data = {'week': forecast_dates.tolist(), 'forecasted_sales': forecasted_sales.tolist()}

    # Return the forecasted sales
    return jsonify(forecast_data)


@app.route('/getSalesData', methods=['GET'])
def get_sales():
    # Define the start and end dates of the dataset
    start_date = pd.to_datetime('2020-01-01')
    end_date = pd.to_datetime('2023-12-31')

    # Generate a sequence of dates from start to end at a weekly frequency
    dates = pd.date_range(start=start_date, end=end_date, freq='W')

    # Generate random sales data
    sales = np.random.randint(low=100, high=1000, size=len(dates))

    # Create a DataFrame with the dates and sales
    df = pd.DataFrame({'week': dates, 'sales': sales})

    # Convert the week column to string format for JSON serialization
    df['week'] = df['week'].dt.strftime('%Y-%m-%d')

    # Output the sales data
    sales_data = df.to_dict('records')
    print(sales_data)
    return sales_data


@app.route('/learn', methods=['POST'])
def learn():
    # Load the data
    data = pd.read_csv('analytics_data.csv')

    # Define the features and target variable
    features = ['purchase_amount', 'click_count', 'conversion_rate']
    target = 'conversion_rate'

    # Split the data into training and testing sets
    X_train = data[features][:1000]
    y_train = data[target][:1000]
    X_test = data[features][1000:]
    y_test = data[target][1000:]

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Print the feature weights
    print("Feature weights:", model.coef_)

    # Determine which features to increase or decrease based on the feature weights
    action = ''
    for i, feature in enumerate(features):
        weight = model.coef_[i]
        if weight > 0.1:
            action = "Increase the weight of {feature} to {weight}"
        elif weight < -0.1:
            action = "Decrease the weight of {feature} to {weight}"
        else:
            action = "The weight of {feature} is optimal"
    return action

def get_seasonal_plots(store):
    fetch_file_from_gcs(store=store,destination_path='good_csv.csv')
    df = pd.read_csv('good_csv.csv')
    df.set_index(df['Date'], inplace=True)
    return df

@app.route('/seasonal_plots', methods=['GET'])
@cross_origin(supports_credentials=True)
def plotSeasonalDecompose():
    store = request.args.get('storeId')
    data = get_seasonal_plots(store=store)
    x = data['Sales']
    result = seasonal_decompose(x, model='additive', period=7)
    fig = make_subplots(rows=4, cols=1)
    
    fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.observed, mode='lines', name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines'), row=3, col=1)
    fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines'), row=4, col=1)
    
    fig.update_layout(
        height=1800,
        width=1000,
        font_family="Calibri",
        template="simple_white",
        barmode="group",
        uniformtext_mode="hide",
        uniformtext_minsize=20,
        showlegend=True,
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    
    fig.update_yaxes(title_text="<b>Observed(INR)</b>", row=1, col=1)
    fig.update_yaxes(title_text="<b>Trend</b>", row=2, col=1)
    fig.update_yaxes(title_text="<b>Seasonal</b>", row=3, col=1)
    fig.update_yaxes(title_text="<b>Residuals</b>", row=4, col=1)
    
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        title_font=dict(family="Calibri", size=15, color="#222222"),
        tickfont=dict(family="Calibri", size=15, color="#222222"),
        showgrid=False,
        title_standoff=35,
    )

    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        title_font=dict(family="Calibri", size=15, color="#222222"),
        tickfont=dict(family="Calibri", size=15, color="#222222"),
        showgrid=False,
        title_standoff=35,
    )
    
    fig.update_layout(
        title={'y': 0.98, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}
    )
    
    html_string = pio.to_html(fig, full_html=False)
    return html_string

@app.route('/train_model', methods=['POST'])
@cross_origin(supports_credentials=True)
def train_model():
    store = request.args.get('storeId')
    fetch_file_from_gcs(store=store,destination_path='good_csv.csv')
    df=pd.read_csv('good_csv.csv')
    validation_data = df.iloc[-30:]
    # Define parameter combinations for grid search
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    P_values = [0, 1, 2]
    D_values = [0, 1]
    Q_values = [0, 1, 2]
    parameter_combinations = list(product(p_values, d_values, q_values, P_values, D_values, Q_values))

    best_mse = float('inf')
    best_params = None

    # Iterate over parameter combinations
    for params in parameter_combinations:
        p, d, q, P, D, Q = params


        # Fit SARIMA model
        model = sm.tsa.SARIMAX(df['Sales'], order=(p, d, q), seasonal_order=(P, D, Q, 12),initialization='approximate_diffuse')
        result = model.fit()

        # Forecast using the validation set
        forecast = result.get_forecast(steps=len(validation_data))
        forecasted_sales = forecast.predicted_mean

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(validation_data['Sales'], forecasted_sales)

        # Update the best parameters if the MSE improves
        if mse < best_mse:
            best_mse = mse
            best_params = params

    # Fit the final model with the best parameters
    p, d, q, P, D, Q = best_params
    model = sm.tsa.SARIMAX(df['Sales'], order=(p, d, q), seasonal_order=(P, D, Q, 12))
    final_result = model.fit(maxiter=5)
    joblib.dump(final_result , 'final_model')
    uplaod_to_gcs('final_model',store)
    return True
   

def uplaod_to_gcs(file,store):
    bucket_name = 'staging-bikayi.appspot.com'
    blob_name = f'trends-engine/{store}/{file}'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file)
    print(f"File {file} uploaded to {bucket_name}/{blob_name}/{'store'}.")
    return f"File {file} uploaded to {bucket_name}/{blob_name}/{'store'}."

@app.route('/predict', methods=['GET'])
@cross_origin(supports_credentials=True)
def predict_sales():
    store = request.args.get('storeId')
    """Fetches a CSV file from Google Cloud Storage."""

    destination_path = 'final_model'

    fetch_file_from_gcs(store=store,destination_path=destination_path)
    fetch_file_from_gcs(store=store,destination_path='good_csv.csv')
  
    load_model=joblib.load('final_model')
    df=pd.read_csv('good_csv.csv')
    last_date = df['Date'].iloc[-1]

    # Generate future dates using pd.date_range()
    future_dates = pd.date_range(start=last_date, periods=30, freq='7D')[1:]

    # Create a DataFrame with the future dates
    future_df = pd.DataFrame({'Date': future_dates, 'Sales': None, 'forecast': None})

    # Concatenate the original DataFrame with the future DataFrame
    ultimate = pd.concat([df, future_df], ignore_index=True)
    ultimate['Date'] = pd.to_datetime(ultimate['Date']) 
    ultimate['forecast'] = load_model.predict(start = len(df), end = (len(df)+30), dynamic= True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ultimate['Date'], y=ultimate['Sales'], mode='lines', name='Sales'))
    fig.add_trace(go.Scatter(x=ultimate['Date'], y=ultimate['forecast'], mode='lines', name='Forecast'))
    fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Sales (INR)'
    )
    fig.update_layout(height=500, width=1000)
    fig.update_layout(title={'y':0.98,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'})
    html_string = pio.to_html(fig, full_html=False)
    return html_string

def fetch_file_from_gcs(store,destination_path):
    bucket_name = 'staging-bikayi.appspot.com'
    file_name = f'trends-engine/{store}/{destination_path}'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.download_to_filename(destination_path)
    print(f"File {destination_path} downloaded to {destination_path}.")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(port=port, host='0.0.0.0')