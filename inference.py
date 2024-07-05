# inference.py
import numpy as np
from models import DeGrootModel
from models import FriedkinModel
from models import HegselmannKrauseModel
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
stubbornness = [0.1, 0.2, 0.1]
epsilon = 0.2

# Создание экземпляров моделей
models = {
    'DeGroot': DeGrootModel(initial_state, influence_matrix),
    'Friedkin': FriedkinModel(initial_state, influence_matrix, stubbornness),
    'Hegselmann-Krause': HegselmannKrauseModel(initial_state, epsilon)
}

# Создание Dash-приложения
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'DeGroot', 'value': 'DeGroot'},
            {'label': 'Friedkin', 'value': 'Friedkin'},
            {'label': 'Hegselmann-Krause', 'value': 'Hegselmann-Krause'}
        ],
        value='DeGroot'
    ),
    dcc.Graph(id='model-graph'),
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
    Output('model-graph', 'figure'),
    Input('model-dropdown', 'value'),
    Input('n-slider', 'value')
)
def update_graph(selected_model, n):
    model = models[selected_model]
    states_up_to_n = model.generate_states_up_to_n(n)
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
            title=f'Состояния модели {selected_model} до шага {n}',
            xaxis={'title': 'Шаги'},
            yaxis={'title': 'Состояние'},
            hovermode='closest'
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)