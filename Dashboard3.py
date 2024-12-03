# Importación de librerías
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Cargar datos
data = pd.read_csv('Datos_limpios_nariño.csv')

# Registrar y cargar el modelo con métrica personalizada
custom_objects = {'mse': MeanSquaredError(name='mse')}
model_path = "modelo_saber11.h5"
model = load_model(model_path, custom_objects=custom_objects)

# Compilar el modelo después de cargarlo
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mse'])

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

# Estilos CSS personalizados
app.layout = html.Div([
    # Encabezado
    html.Div([
        html.H1("Dashboard Pruebas Saber 11", style={'color': '#343a40', 'fontFamily': 'Arial'}),
        html.P("Explora cómo las características del colegio y el hogar influyen en el rendimiento académico.", 
               style={'color': '#6c757d', 'fontFamily': 'Arial', 'fontSize': '16px'}),
    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#e9ecef'}),

    # Contenedor principal
    html.Div([
        # Panel de filtros
        html.Div([
            html.H3("Filtros", style={'color': '#343a40', 'fontFamily': 'Arial'}),
            html.Label("Área de Ubicación", style={'color': '#495057', 'fontFamily': 'Arial'}),
            dcc.Dropdown(id='area-filter', options=area_options, multi=True, 
                         placeholder="Selecciona una o más áreas..."),
            
            html.Label("Jornada", style={'color': '#495057', 'fontFamily': 'Arial'}),
            dcc.Dropdown(id='jornada-filter', options=jornada_options, multi=True, 
                         placeholder="Selecciona una o más jornadas..."),
            
            html.Label("Educación de la Madre", style={'color': '#495057', 'fontFamily': 'Arial'}),
            dcc.Dropdown(id='educacionmadre-filter', options=educacionmadre_options, multi=True, 
                         placeholder="Selecciona una opción..."),
            
            html.Hr(),
            
            html.H4("Predicción del Puntaje Global", style={'color': '#343a40', 'fontFamily': 'Arial'}),
            dcc.Input(id='prediction-input', type='number', placeholder='Introduce un valor', 
                      style={'width': '100%', 'padding': '10px', 'marginBottom': '10px'}),
            html.Button('Predecir', id='predict-button', n_clicks=0, 
                        style={'width': '100%', 'padding': '10px', 'backgroundColor': '#007bff', 
                               'color': 'white', 'border': 'none', 'cursor': 'pointer'}),
            html.Div(id='prediction-output', style={'marginTop': '10px', 'color': '#495057'}),
        ], style={'width': '30%', 'float': 'left', 'padding': '20px', 'backgroundColor': '#f8f9fa', 
                  'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'borderRadius': '8px'}),
        
        # Panel de gráficos
        html.Div([
            dcc.Graph(id='puntaje-global-graph'),
            dcc.Graph(id='areas-especificas-graph'),
        ], style={'width': '65%', 'float': 'right', 'padding': '20px'}),
    ]),
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
                        title="Distribución del Puntaje Global por Jornada",
                        color_discrete_sequence=px.colors.qualitative.Pastel)
    fig2 = px.scatter(filtered_data, x='fami_tienecomputador', y='punt_global',
                      color='fami_tieneinternet', 
                      title="Relación entre Recursos Educativos y Puntaje Global",
                      color_discrete_sequence=px.colors.qualitative.Bold)
    
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
