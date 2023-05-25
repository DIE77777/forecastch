# First, you'll need to install the required libraries
#!pip install fbprophet pmdarima

import sys
import pandas as pd
from fbprophet import Prophet
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import time
import base64

print("Python version")
print(sys.version)
print("Pandas version")
print(pd.__version__)
print("Scikit-learn version")
print(sklearn.__version__)
print("Statsmodels version")
print(sm.__version__)
print("NumPy version")
print(np.__version__)
print("Streamlit version")
print(st.__version__)

# For libraries that don't offer a .__version__ attribute, you may need to use an alternative approach to get the version. For instance, you can use the pkg_resources module to get the version of fbprophet:
import pkg_resources
print("fbprophet version")
print(pkg_resources.get_distribution("fbprophet").version)

# pmdarima doesn't have a .__version__ attribute, but we can get its version from the installed packages.
print("pmdarima version")
print(pkg_resources.get_distribution("pmdarima").version)


# define the start_timestart

start_time = time.time()

# create functiom to get_table_download_link in xlsx format

def get_table_download_link(df):
  
      """Generates a link allowing the data in a given panda dataframe to be downloaded
  
      in:  dataframe
  
      out: href string
  
      """
  
      csv = df.to_csv(index=False)
  
      b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
  
      href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
  
      return href

# create a function to get_table_download_link in xlsx format

def get_table_download_link_xlsx(df):
  
      """Generates a link allowing the data in a given panda dataframe to be downloaded
  
      in:  dataframe
  
      out: href string
  
      """
  
      towrite = io.BytesIO()
  
      downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
  
      towrite.seek(0)  # reset pointer
  
      b64 = base64.b64encode(towrite.read()).decode()
  
      href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="extract.xlsx">Download xlsx file</a>'
  
      return href

# create app title

st.title('Forecasting App')

# insert a logo

#st.image('logo.jpg', use_column_width=True)


# create a slider bar to select BETWEEN MARCAS DE CANAL OR TEAM SOLUTIONS

option = st.sidebar.selectbox('Select the option',('CONSUMO HOGAR','TEAM SOLUTIONS'))

# create the info message with the propose of the app

st.info('This app is designed to forecast the sales of the selected UN')

# create  subheader
st.subheader('Upload your data')

# up load the file to the app

uploaded_file = st.file_uploader("Choose a file")

# create a botton to run the app

run = st.button('Run')


# if the file is not empty, load it into a pandas dataframe

if uploaded_file is not None and run == True:

    # read VENTAS_CH.xlsx file

    #df=pd.read_excel('VENTAS_CH_OCT2022.xlsx', sheet_name='cargue_r')

    # Load the data into a Pandas dataframe
    
    df = pd.read_excel(uploaded_file)  

    # UPPER CASE the column YEAR and MONTH

    df['MES'] = df['MES'].str.upper()

    # replace the MES column with the corresponding month umbers

    df['MES'] = df['MES'].replace({'ENE': 1, 'FEB': 2, 'MAR': 3, 'ABR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AGO': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DIC': 12})

    # make a date column from the year and month columns

    df['ds'] = df['AÑO'].astype(str) + '-' + df['MES'].astype(str)+ '-01'

    # convert the ds column to a datetime object

    df['ds'] = pd.to_datetime(df['ds'])

    # drop the year and month columns

    df.drop(['AÑO', 'MES'], axis=1, inplace=True)

     # create a key with the produuct,cliente an segmento columns

    df['producto'] = df['Canal_Agrupado'].astype(str) + df['COD_PRODUCTO'].astype(str) 

    # eliminate the negative values in the venta column

    df['VENTA NETA (TON)'] = df['VENTA NETA (TON)'].apply(lambda x: 0 if x < 0 else x)

    df_apertura= df.copy()

    # agrupate the data by prducto,Canal_Agrupado,COD_PRODUCTO, AÑO and MES columns

    df = df.groupby(['producto','Canal_Agrupado','COD_PRODUCTO', 'ds'])['VENTA NETA (TON)'].sum().reset_index()

    df_export = df.copy()

    # rename the venta column to y

    df.rename(columns={'VENTA NETA (TON)': 'y'}, inplace=True)

    df ['y'] =df ['y'].astype(float)

    # fit the y negative values to 0

    df['y'] = df['y'].apply(lambda x: 0 if x < 0 else x)

    # store the list of products than have less than 10 observations

    product_filter= df.groupby('producto').filter(lambda x: len(x) <= 10).producto.unique()

    # only select the products with more than 10 observations

    df = df.groupby('producto').filter(lambda x: len(x) > 10)

    # Group the data by product
    grouped = df.groupby('producto')

    # select key, ds and y columns

    df = df[['producto', 'ds', 'y']]

    # construct a dataframe to store the forecast results

    forecast_df = pd.DataFrame()

    # Estas dos lineas son pruebas

    # only take 10 product of the df#######################

    #df = df[df['producto'].isin(df['producto'].unique()[:30])]

    # Group the data by product
    #grouped = df.groupby('producto')

    # select the same products in df and df_export

    #df_export = df_export[df_export['producto'].isin(df['producto'].unique())]

    # select the same products in df and df_apertura

    #df_apertura= df_apertura[df_apertura['producto'].isin(df['producto'].unique())]

    #######################################################

    # Iterate over the products 
    for name, group in grouped:

      # Split the data into a training set and a test set
      train = group.iloc[:int(0.8*len(group))]
      test = group.iloc[int(0.8*len(group)):]

      # Fit the FB Prophet model to the training data
      m1 = Prophet()
      m1.fit(train)  

      # Fit the pmdarima model to the training data
      m2 = auto_arima(train['y'], error_action='ignore', suppress_warnings=True)

      # fit the mean average model to the training data

      m3 = sm.tsa.SimpleExpSmoothing(train['y']).fit(smoothing_level=0)

      # fit the naive model to the forecast 

      m4 = sm.tsa.SimpleExpSmoothing(train['y']).fit(smoothing_level=1,optimized=False)

      # fit the random forest model to the training data

      m5 = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0).fit(train['ds'].values.reshape(-1, 1), train['y'])

              
      # Make forecasts on the test data using both models
      
      forecast1 = m1.predict(test)
      forecast2 = m2.predict(n_periods=len(test))
      forecast3 = m3.predict(start=test.index[0], end=test.index[-1])
      forecast4 = m4.predict(start=test.index[0], end=test.index[-1])
      forecast5 = m5.predict(test['ds'].values.reshape(-1, 1))


      # Evaluate the accuracy of the forecasts using mean absolute error (MAE)
      
      mae1 = mean_absolute_error(test['y'], forecast1['yhat'])
      mae2 = mean_absolute_error(test['y'], forecast2)
      mae3 = mean_absolute_error(test['y'], forecast3)
      mae4 = mean_absolute_error(test['y'], forecast4)
      mae5 = mean_absolute_error(test['y'], forecast5)

      # Print the MAE for both models
      print(f'MAE for FB Prophet ({name}): {mae1}')
      print(f'MAE for pmdarima ({name}): {mae2}')
      print(f'MAE for mean average ({name}): {mae3}')
      print(f'MAE for naive ({name}): {mae4}')
      print(f'MAE for random forest ({name}): {mae5}')
    

      # Choose the model with the lowest MAE and print the name of the product
      if mae1 < mae2:
        print(f'FB Prophet is better for {name}')

        # apply the prophet model to the entire dataset
        m1 = Prophet()
        m1.fit(group)

        # Make a forecast for the next 12 months
        future = m1.make_future_dataframe(periods=12, freq='M')
        forecast = m1.predict(future)

        # take only the last 12 months of the forecast

        forecast = forecast.iloc[-12:]

        # Add the forecast to the forecast_df dataframe
        forecast['producto'] = name
        forecast['pronostico'] = forecast['yhat']    

        
        # aggregate the name of the model to the forecast dataframe
        forecast['modelo'] = 'FB Prophet'

        # select the columns we want to keep
        forecast = forecast[['producto', 'pronostico','modelo']]
        
        forecast_df = forecast_df.append(forecast)

      elif mae3 < mae2:

        print(f'mean average is better for {name}')

        # apply the mean average model to the entire dataset
        m3 = sm.tsa.SimpleExpSmoothing(group['y']).fit()

        # Make a forecast for the next 12 months

        forecast = m3.forecast(12)         
        
        # Add the forecast to the forecast_df dataframe
        forecast = pd.DataFrame(forecast)
        forecast['producto'] = name
        forecast['pronostico'] = forecast[0]    
        

        # aggregate the name of the model to the forecast dataframe

        forecast['modelo'] = 'mean average'

        # select the columns we want to keep
        forecast = forecast[['producto', 'pronostico', 'modelo']]
        forecast_df = forecast_df.append(forecast)

      elif mae4 < mae2:

        print(f'naive is better for {name}')

        # apply the naive model to the entire dataset
        m4 = sm.tsa.SimpleExpSmoothing(group['y']).fit(smoothing_level=1,optimized=False)

        # Make a forecast for the next 12 months

        forecast = m4.forecast(12)

          
        # Add the forecast to the forecast_df dataframe
        forecast = pd.DataFrame(forecast)
        forecast['producto'] = name
        forecast['pronostico'] = forecast[0]    

        # aggregate the name of the model to the forecast dataframe
        forecast['modelo'] = 'naive'
        
        # select the columns we want to keep
        forecast = forecast[['producto', 'pronostico', 'modelo']]

        forecast_df = forecast_df.append(forecast)

      elif mae5 < mae2:
        
            print(f'random forest is better for {name}')
        
            # apply the random forest model to the entire dataset
            m5 = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0).fit(group['ds'].values.reshape(-1, 1), group['y'])
        
            # Make a forecast for the next 12 months
            forecast = m5.predict(pd.date_range(start='2022-01-31', periods=12, freq='M').values.reshape(-1, 1))               
        
            # Add the forecast to the forecast_df dataframe
            forecast = pd.DataFrame(forecast)
            forecast['producto'] = name
            forecast['pronostico'] = forecast[0]   
            
            # aggregate the name of the model to the forecast dataframe
            forecast['modelo'] = 'random forest'
            
            # select the columns we want to keep
            forecast = forecast[['producto', 'pronostico', 'modelo']]
        
            forecast_df = forecast_df.append(forecast)
            
        
      else:
        print(f'pmdarima is better for {name}')

        # apply the pmdarima model to the entire dataset
        m2 = auto_arima(group['y'], error_action='ignore', suppress_warnings=True)

        # Make a forecast for the next 12 months
        forecast = m2.predict(n_periods=12)

        # Add the forecast to the forecast_df dataframe
        forecast = pd.DataFrame(forecast)
        forecast['producto'] = name
        forecast['pronostico'] = forecast[0]    
        

        # aggregate the name of the model to the forecast dataframe
        forecast['modelo'] = 'pmdarima'
        
        # select the columns we want to keep
        forecast = forecast[['producto', 'pronostico','modelo']]

        forecast_df = forecast_df.append(forecast)
    # Print the forecast_df dataframe

    print(forecast_df)

    #remplace the negative values with 0

    forecast_df['pronostico'] = forecast_df['pronostico'].apply(lambda x: 0 if x < 0 else x)

    # enumerar de 1 a 12 cada pronostico para cada producto y llamarlo periodo

    forecast_df['periodo'] = forecast_df.groupby('producto').cumcount() + 1


    # transform the forecast_df period in string

    forecast_df['periodo'] = forecast_df['periodo'].astype(str)

    # transform the forecast_df dataframe in a wide format

    forecast_wide = forecast_df.pivot_table(index=['producto'], columns='periodo', values='pronostico')

    # estimate the average  of the all the periods of the forecast_wide dataframe

    forecast_wide['avg_fct'] = forecast_wide.iloc[:, :].mean(axis=1)

    # Transform the df_export in the rows 'UN', 'COD. PRODUCTO', 'PRODUCTO', 'SEGMENTO', 'COD CLIENTE', 'CLIENTE' and the columns 'ds' and 'VENTA NETA (TON)' 

    df_export_wide = df_export.pivot_table(index=['producto', 'Canal_Agrupado', 'COD_PRODUCTO'], columns='ds', values='VENTA NETA (TON)')

    #  repit the same index in the df_export_wide

    df_export_wide = df_export_wide.reset_index()

    # fill the nan values with 0

    df_export_wide.fillna(0, inplace=True)

    # estimate the average of the last 3 months of the df_export_wide dataframe taking acount all values 

    df_export_wide['avg'] = df_export_wide.iloc[:, -3:].mean(axis=1)

    df_export_wide['avg'].fillna(0, inplace=True)

    # create a key to join the df_export_wide with the forecast_df dataframe

    df_export_wide['key'] = df_export_wide['producto'].astype(str)
    # fill the nan values with 0

    df_export_wide.fillna(0, inplace=True)

    # join the df_export_wide with the forecast_df dataframe

    df_export_wide = df_export_wide.merge(forecast_wide, left_on='key', right_on='producto', how='left')


    # if the avg columns in df_export_wide is not 0 and the forecast_wide columns are 0  or null then replace the forecast_wide columns with the avg value

    for i in df_export_wide.columns[-13:]:

      df_export_wide[i] = np.where((df_export_wide['avg'] != 0) & (df_export_wide[i].isnull() | (df_export_wide[i] == 0)), df_export_wide['avg'], df_export_wide[i])


    # if the avg_fct is 30% higher or lower than the avg then replace the forecast_wide columns with the avg value

    for i in df_export_wide.columns[-13:]:
      df_export_wide[i] = np.where((df_export_wide['avg_fct'] >= df_export_wide['avg'] * 1.4) | (df_export_wide['avg_fct'] <= df_export_wide['avg'] * 0.6), df_export_wide['avg'], df_export_wide[i])

    # For teh products in product_filter replace the forecast_wide columns with the avg value

    for i in df_export_wide.columns[-13:]:
      df_export_wide[i] = np.where(df_export_wide['COD_PRODUCTO'].isin(product_filter), df_export_wide['avg'], df_export_wide[i])

    # rename forecast columns

    df_export_wide.rename(columns={'1': 'PVO_1', '2': 'PVO_2', '3': 'PVO_3', '4': 'PVO_4', '5': 'PVO_5', '6': 'PVO_6', '7': 'PVO_7', '8': 'PVO_8', '9': 'PVO_9', '10': 'PVO_10', '11': 'PVO_11', '12': 'PVO_12', '13': 'PVO_13', '14': 'PVO_14'}, inplace=True)
       
 
try:

    st.write(df_export_wide)

    # link to download the df_export_wide dataframe

    st.subheader('Descargar datos')

    st.markdown(get_table_download_link_xlsx(df_export_wide), unsafe_allow_html=True)

    #read the df_export_wide dataframe

    #df_export_wide = pd.read_excel('forecast.xlsx')

    # filter df_apertura  with ds >= 2022-10-01

    df_apertura = df_apertura[df_apertura['ds'] >= '2022-06-01']

    # agroup df_apertura by producto and sum the VENTA NETA (TON) column

    df_apertura = df_apertura.groupby(['producto', 'Canal_Agrupado', 'Nombre Canal','Categoria', 'Marca','COD_PRODUCTO', 'Descripción_Team','Cliente', 'Director Canal'])['VENTA NETA (TON)'].sum().reset_index()

    

    # estimate  the participation of the product in the total sales of the client

    df_apertura['participacion'] = df_apertura['VENTA NETA (TON)'] / df_apertura.groupby('producto')['VENTA NETA (TON)'].transform('sum')    

    
    # select on df_export_wide the columns 'producto', 'PVO_1', 'PVO_2', 'PVO_3', 'PVO_4', 'PVO_5','PVO_6', 'PVO_7', 'PVO_8', 'PVO_9', 'PVO_10', 'PVO_11', 'PVO_12'

    df_export_wide = df_export_wide[['producto', 'PVO_1', 'PVO_2', 'PVO_3', 'PVO_4', 'PVO_5','PVO_6', 'PVO_7', 'PVO_8', 'PVO_9', 'PVO_10', 'PVO_11', 'PVO_12']]

    # join the df_apertura with the df_export_wide dataframe

    df_export_wide_aper = df_apertura.merge(df_export_wide, left_on='producto', right_on='producto', how='left')

    # Multiplicate the columns that contain the PVO word in the name of the column by the participacion column

    df_export_wide_aper['forecast_1'] = df_export_wide_aper['PVO_1'] * df_export_wide_aper['participacion']
    df_export_wide_aper['forecast_2'] = df_export_wide_aper['PVO_2'] * df_export_wide_aper['participacion']
    df_export_wide_aper['forecast_3'] = df_export_wide_aper['PVO_3'] * df_export_wide_aper['participacion']
    df_export_wide_aper['forecast_4'] = df_export_wide_aper['PVO_4'] * df_export_wide_aper['participacion']
    df_export_wide_aper['forecast_5'] = df_export_wide_aper['PVO_5'] * df_export_wide_aper['participacion']
    df_export_wide_aper['forecast_6'] = df_export_wide_aper['PVO_6'] * df_export_wide_aper['participacion']
    df_export_wide_aper['forecast_7'] = df_export_wide_aper['PVO_7'] * df_export_wide_aper['participacion']
    df_export_wide_aper['forecast_8'] = df_export_wide_aper['PVO_8'] * df_export_wide_aper['participacion']
    df_export_wide_aper['forecast_9'] = df_export_wide_aper['PVO_9'] * df_export_wide_aper['participacion']
    df_export_wide_aper['forecast_10'] = df_export_wide_aper['PVO_10'] * df_export_wide_aper['participacion']
    df_export_wide_aper['forecast_11'] = df_export_wide_aper['PVO_11'] * df_export_wide_aper['participacion']
    df_export_wide_aper['forecast_12'] = df_export_wide_aper['PVO_12'] * df_export_wide_aper['participacion']
   

    # create a link to download the df_export_wide dataframe

    st.subheader('Descargar datos')

    st.markdown(get_table_download_link_xlsx(df_export_wide_aper), unsafe_allow_html=True)


     
    st.write('Tiempo de ejecución: ', time.time() - start_time, 'segundos')

    st.balloons()

    st.write('Proceso finalizado')

except:
    
        st.write('No hay datos para mostrar')




















    








    









   
  