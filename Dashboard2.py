# Importación de librerías
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Cargar datos
data = pd.read_csv('Datos_limpios_nariño.csv')

# Registrar y cargar el modelo con métrica personalizada
custom_objects = {'mse': MeanSquaredError(name='mse')}
model_path = "modelo_saber11.h5"
model = load_model(model_path, custom_objects=custom_objects)

# Preprocesamiento
categorical_cols = ['cole_area_ubicacion', 'cole_bilingue', 'cole_calendario', 'cole_caracter',
                    'cole_jornada', 'fami_educacionmadre', 'fami_educacionpadre', 
                    'fami_tienecomputador', 'fami_tieneinternet']
numerical_cols = ['punt_global']

# Ajustar preprocesadores
encoder = OneHotEncoder(sparse_output=False).fit(data[categorical_cols])
scaler = StandardScaler().fit(data[numerical_cols])

# Opciones para filtros
area_options = [{'label': area, 'value': area} for area in data['cole_area_ubicacion'].unique()]
jornada_options = [{'label': jornada, 'value': jornada} for jornada in data['cole_jornada'].unique()]
educacionmadre_options = [{'label': edu, 'value': edu} for edu in sorted(data['fami_educacionmadre'].unique())]

# Crear la aplicación Dash
app = dash.Dash(__name__)

# Layout del Dashboard
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Dashboard Pruebas Saber 11"),
        html.P("Analiza cómo las características del colegio y el hogar afectan los resultados."),
    ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f8f9fa'}),
    
    # Panel de filtros
    html.Div([
        html.H3("Filtros"),
        html.Label("Área de Ubicación"),
        dcc.Dropdown(id='area-filter', options=area_options, multi=True),
        
        html.Label("Jornada"),
        dcc.Dropdown(id='jornada-filter', options=jornada_options, multi=True),
        
        html.Label("Educación de la Madre"),
        dcc.Dropdown(id='educacionmadre-filter', options=educacionmadre_options, multi=True),
        
        html.Label("Predicción del Puntaje Global"),
        dcc.Input(id='prediction-input', type='number', placeholder='Introduce el puntaje global esperado'),
        html.Button('Predecir', id='predict-button', n_clicks=0),
        html.Div(id='prediction-output'),
    ], style={'width': '25%', 'float': 'left', 'padding': '10px', 'backgroundColor': '#f1f1f1'}),
    
    # Panel de gráficos
    html.Div([
        dcc.Graph(id='puntaje-global-graph'),
        dcc.Graph(id='areas-especificas-graph'),
    ], style={'width': '70%', 'float': 'right', 'padding': '10px'}),
])

# Callback para actualizar gráficos
@app.callback(
    [Output('puntaje-global-graph', 'figure'),
     Output('areas-especificas-graph', 'figure')],
    [Input('area-filter', 'value'),
     Input('jornada-filter', 'value'),
     Input('educacionmadre-filter', 'value')]
)
def update_graphs(area, jornada, educacionmadre):
    filtered_data = data.copy()
    if area:
        filtered_data = filtered_data[filtered_data['cole_area_ubicacion'].isin(area)]
    if jornada:
        filtered_data = filtered_data[filtered_data['cole_jornada'].isin(jornada)]
    if educacionmadre:
        filtered_data = filtered_data[filtered_data['fami_educacionmadre'].isin(educacionmadre)]
    
    fig1 = px.histogram(filtered_data, x='punt_global', color='cole_jornada',
                        title="Distribución del Puntaje Global por Jornada")
    fig2 = px.scatter(filtered_data, x='fami_tienecomputador', y='punt_global',
                      color='fami_tieneinternet', 
                      title="Relación entre Recursos Educativos y Puntaje Global")
    
    return fig1, fig2

# Callback para predicción
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('prediction-input', 'value')]
)
def predict_puntaje(n_clicks, input_value):
    if n_clicks > 0 and input_value is not None:
        input_data = np.array([[input_value]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        return f"Puntaje predicho: {prediction[0][0]:.2f}"
    return "Introduce un valor y presiona predecir."

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)

