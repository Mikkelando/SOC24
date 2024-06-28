import numpy as np
from models import DeGrootModel
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

initial_state = [0.6, 0.4, 0.7]
influence_matrix = [
    [0.5, 0.3, 0.2],
    [0.2, 0.5, 0.3],
    [0.3, 0.2, 0.5]
]

# Создание экземпляра модели Дегрута
degroot_model = DeGrootModel(initial_state, influence_matrix)

# Генерация состояний от начального до 10-го шага
n = 10
states_up_to_n = degroot_model.generate_states_up_to_n(n)

# Преобразование состояний для графика
time_steps = list(range(n + 1))
states = np.array(states_up_to_n).T

# Создание Dash-приложения
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='degroot-graph'),
    dcc.Slider(
        id='n-slider',
        min=1,
        max=50,
        step=1,
        value=10,
        marks={i: str(i) for i in range(1, 51)}
    )
])

@app.callback(
    Output('degroot-graph', 'figure'),
    Input('n-slider', 'value')
)
def update_graph(n):
    states_up_to_n = degroot_model.generate_states_up_to_n(n)
    time_steps = list(range(n + 1))
    states = np.array(states_up_to_n).T
    
    traces = []
    for i in range(states.shape[0]):
        traces.append(go.Scatter(
            x=time_steps,
            y=states[i],
            mode='lines+markers',
            name=f'Agent {i+1}'
        ))
    
    return {
        'data': traces,
        'layout': go.Layout(
            title=f'Состояния модели Дегрута до шага {n}',
            xaxis={'title': 'Шаги'},
            yaxis={'title': 'Состояние'},
            hovermode='closest'
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)
