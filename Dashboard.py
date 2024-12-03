from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px

data = pd.read_csv("C:\\Users\\oscar\\Desktop\\Uniandes\\10 Semestre\\Analitica computacional para la toma de decisiones\\Proyecto\\Proyecto 3\\Datos_limpios_nariño.csv")


# Crear la aplicación Dash
app = Dash(__name__)

# Layout del Dashboard
app.layout = html.Div([
    html.H1("Dashboard de Análisis - Pruebas Saber 11", style={'textAlign': 'center'}),

    # Filtros
    html.Div([
        html.Label("Filtrar por área de ubicación:"),
        dcc.Dropdown(
            id='filtro_area',
            options=[{'label': area, 'value': area} for area in data['cole_area_ubicacion'].unique()],
            value=None,
            placeholder="Seleccione un área"
        ),
        html.Label("Filtrar por bilingüismo:"),
        dcc.Dropdown(
            id='filtro_bilingue',
            options=[{'label': 'Sí', 'value': 'S'}, {'label': 'No', 'value': 'N'}],
            value=None,
            placeholder="Seleccione una opción"
        ),
    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    # Resultados y gráficos
    html.Div([
        dcc.Graph(id='grafico_puntaje_global'),
        dcc.Graph(id='grafico_condiciones_hogar')
    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    # Instrucciones
    html.Div([
        html.H3("Instrucciones"),
        html.P("1. Utilice los filtros a la izquierda para explorar los datos según características específicas."),
        html.P("2. Los gráficos reflejarán los datos filtrados y mostrarán relaciones clave."),
        html.P("3. Pase el cursor sobre los gráficos para obtener detalles interactivos.")
    ], style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid black'})
])

# Callbacks para interactividad
@app.callback(
    Output('grafico_puntaje_global', 'figure'),
    Output('grafico_condiciones_hogar', 'figure'),
    Input('filtro_area', 'value'),
    Input('filtro_bilingue', 'value')
)
def actualizar_graficos(area, bilingue):
    # Filtrar datos
    datos_filtrados = data.copy()
    if area:
        datos_filtrados = datos_filtrados[datos_filtrados['cole_area_ubicacion'] == area]
    if bilingue:
        datos_filtrados = datos_filtrados[datos_filtrados['cole_bilingue'] == bilingue]
    
    # Gráfico 1: Puntaje global promedio por característica del colegio
    fig1 = px.bar(datos_filtrados, x='cole_jornada', y='punt_global', color='cole_caracter',
                  title="Puntaje Global Promedio por Jornada y Naturaleza del Colegio",
                  labels={'cole_jornada': 'Jornada', 'punt_global': 'Puntaje Global'})
    
    # Gráfico 2: Relación entre condiciones del hogar y puntaje global
    fig2 = px.scatter(datos_filtrados, x='fami_tienecomputador', y='punt_global',
                      color='fami_educacionmadre', 
                      title="Relación entre Recursos del Hogar y Puntaje Global",
                      labels={'fami_tienecomputador': 'Tiene Computador', 'punt_global': 'Puntaje Global'})
    
    return fig1, fig2

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run_server(debug=True)
