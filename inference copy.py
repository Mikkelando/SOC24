# inference.py
import numpy as np
from models import DeGrootModel
from models import FriedkinModel
from models import HegselmannKrauseModel
from models.degroot_inf import InfiniteDeGrootModel
from models.friedkin_inf import InfiniteFriedkinModel
from models.hegselmann_inf import InfiniteHegselmannKrauseModel
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

delta = 0.1
epsilon = 0.2
K = 2
L1 = 100
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
    'Hegselmann-Krause': HegselmannKrauseModel(initial_state, epsilon),
    'InfiniteDeGroot': InfiniteDeGrootModel(delta, K, L1),
    'InfiniteFriedkin': InfiniteFriedkinModel(delta, K, L1),
    'InfiniteHegselmannKrause': InfiniteHegselmannKrauseModel(delta, epsilon, L1)
}

# Создание Dash-приложения
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'DeGroot', 'value': 'DeGroot'},
            {'label': 'Friedkin', 'value': 'Friedkin'},
            {'label': 'Hegselmann-Krause', 'value': 'Hegselmann-Krause'},
            {'label': 'InfiniteDeGroot', 'value': 'InfiniteDeGroot'},
            {'label': 'InfiniteFriedkin', 'value': 'InfiniteFriedkin'},
            {'label': 'InfiniteHegselmannKrause', 'value': 'InfiniteHegselmannKrause'}
        ],
        value='InfiniteDeGroot'
    ),
    dcc.Graph(id='model-graph'),
    dcc.Slider(
        id='n-slider',
        min=1,
        max=50,
        step=1,
        value=10,
        marks={i: str(i) for i in range(1, 51)}
    ),
    html.Div([
        dcc.Input(
            id='T-input',
            type='number',
            value=10,
            placeholder='Число шагов T'
        ),
        dcc.Input(
            id='K-input',
            type='number',
            value=2,
            placeholder='Параметр K'
        ),
        dcc.Input(
            id='epsilon-input',
            type='number',
            value=0.2,
            placeholder='Порог epsilon'
        )
    ], style={'margin-top': '10px'})
])

@app.callback(
    Output('model-graph', 'figure'),
    Input('model-dropdown', 'value'),
    Input('n-slider', 'value'),
    Input('T-input', 'value'),
    Input('K-input', 'value'),
    Input('epsilon-input', 'value')
)
def update_graph(selected_model, n, T, K, epsilon):
    model = models[selected_model]
    
    if selected_model == 'InfiniteHegselmannKrause':
        model.epsilon = epsilon
        model.adjust_participants(T)
    elif selected_model.startswith('Infinite'):
        model.K = K  # Обновляем значение K для модели
        model.adjust_participants(T)
    
    states_up_to_n = model.generate_states_up_to_n(n)
    time_steps = list(range(n + 1))
    states = np.array(states_up_to_n).T
    
    traces = []
    for i in range(min(states.shape[0], 100)):  # Ограничение на отображение до 100 агентов
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
