import numpy as np
import pandas as pd
from models import DeGrootModel, FriedkinModel, HegselmannKrauseModel
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# Функция для генерации случайной матрицы влияния
def generate_random_matrix(size):
    matrix = np.random.rand(int(size), int(size))
    return matrix / matrix.sum(axis=1)[:, None]

# Функция для генерации случайных начальных значений
def generate_random_initial_state(size):
    return np.random.rand(int(size)).tolist()

# Создание Dash-приложения
app = dash.Dash(__name__)

app.layout = html.Div(
    className='container',
    children=[
        html.Div([
            html.Label('Number of Agents'),
            dcc.Input(id='agent-count-input', value='3', type='number', min=2),
            html.Button('Apply', id='apply-agent-count-btn')
        ]),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'DeGroot', 'value': 'DeGroot'},
                {'label': 'Friedkin', 'value': 'Friedkin'},
                {'label': 'Hegselmann-Krause', 'value': 'Hegselmann-Krause'}
            ],
            value='DeGroot'
        ),
        html.Div([
            html.Label('Initial State (comma separated)'),
            dcc.Input(id='initial-state-input', value='0.6,0.4,0.7', type='text'),
            html.Button('Generate Initial State', id='generate-initial-state-btn'),
            html.Button('Save Initial State', id='save-initial-state-btn')
        ]),
        html.Div([
            html.Label('Influence Matrix (comma separated rows)'),
            dcc.Textarea(
                id='influence-matrix-input',
                value='0.5,0.3,0.2\n0.2,0.5,0.3\n0.3,0.2,0.5',
                style={'width': '100%', 'height': 100}
            ),
            html.Button('Generate Influence Matrix', id='generate-matrix-btn'),
            html.Button('Save Influence Matrix', id='save-matrix-btn')
        ]),
        dcc.Graph(id='model-graph', className='graph'),
        dcc.RangeSlider(
            id='n-slider',
            min=0,
            max=50,
            step=1,
            value=[0, 10],
            marks={i: str(i) for i in range(0, 51)}
        ),
        html.Div(id='save-output', style={'marginTop': 20})
    ]
)

@app.callback(
    [Output('initial-state-input', 'value'),
     Output('influence-matrix-input', 'value')],
    [Input('generate-initial-state-btn', 'n_clicks'),
     Input('generate-matrix-btn', 'n_clicks'),
     Input('apply-agent-count-btn', 'n_clicks')],
    [State('agent-count-input', 'value'),
     State('initial-state-input', 'value'),
     State('influence-matrix-input', 'value')]
)
def update_inputs(n_clicks_init, n_clicks_matrix, n_clicks_apply, agent_count, initial_state_value, influence_matrix_value):
    ctx = dash.callback_context

    if not ctx.triggered:
        return initial_state_value, influence_matrix_value

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'apply-agent-count-btn':
        initial_state = generate_random_initial_state(agent_count)
        influence_matrix = generate_random_matrix(agent_count)
        initial_state_str = ','.join(map(str, initial_state))
        influence_matrix_str = '\n'.join([','.join(map(str, row)) for row in influence_matrix])
        return initial_state_str, influence_matrix_str

    if trigger_id == 'generate-initial-state-btn':
        initial_state = generate_random_initial_state(agent_count)
        initial_state_str = ','.join(map(str, initial_state))
        return initial_state_str, influence_matrix_value

    if trigger_id == 'generate-matrix-btn':
        influence_matrix = generate_random_matrix(agent_count)
        influence_matrix_str = '\n'.join([','.join(map(str, row)) for row in influence_matrix])
        return initial_state_value, influence_matrix_str

    return initial_state_value, influence_matrix_value

@app.callback(
    Output('model-graph', 'figure'),
    [Input('model-dropdown', 'value'),
     Input('initial-state-input', 'value'),
     Input('influence-matrix-input', 'value'),
     Input('n-slider', 'value')]
)
def update_graph(selected_model, initial_state_str, influence_matrix_str, n_range):
    # Преобразование строки начального состояния в список чисел
    initial_state = list(map(float, initial_state_str.split(',')))

    # Преобразование строки матрицы влияния в numpy array
    influence_matrix = np.array(
        [list(map(float, row.split(','))) for row in influence_matrix_str.split('\n')]
    )

    # Создание экземпляра выбранной модели
    if selected_model == 'DeGroot':
        model = DeGrootModel(initial_state, influence_matrix)
    elif selected_model == 'Friedkin':
        stubbornness = [0.1] * len(initial_state)  # Пример жесткости агентов
        model = FriedkinModel(initial_state, influence_matrix, stubbornness)
    else:
        epsilon = 0.2  # Пример параметра для модели Хегсельмана-Краузе
        model = HegselmannKrauseModel(initial_state, epsilon)

    # Генерация состояний
    states_up_to_n = model.generate_states_up_to_n(n_range[1])
    time_steps = list(range(n_range[0], n_range[1] + 1))
    states = np.array(states_up_to_n).T[:, n_range[0]:n_range[1] + 1]

    # Построение графика
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
            title=f'Состояния модели {selected_model} от шага {n_range[0]} до {n_range[1]}',
            xaxis={'title': 'Шаги'},
            yaxis={'title': 'Состояние'},
            hovermode='closest'
        )
    }

@app.callback(
    Output('save-output', 'children'),
    [Input('save-initial-state-btn', 'n_clicks'),
     Input('save-matrix-btn', 'n_clicks')],
    [State('initial-state-input', 'value'),
     State('influence-matrix-input', 'value')]
)
def save_data(n_clicks_initial, n_clicks_matrix, initial_state_value, influence_matrix_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ''
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'save-initial-state-btn':
        initial_state = list(map(float, initial_state_value.split(',')))
        df_initial_state = pd.DataFrame(initial_state, columns=['Initial State'])
        df_initial_state.to_csv('initial_state.csv', index=False)
        return 'Initial state saved to initial_state.csv'
    
    if trigger_id == 'save-matrix-btn':
        influence_matrix = np.array(
            [list(map(float, row.split(','))) for row in influence_matrix_value.split('\n')]
        )
        df_influence_matrix = pd.DataFrame(influence_matrix)
        df_influence_matrix.to_csv('influence_matrix.csv', index=False)
        return 'Influence matrix saved to influence_matrix.csv'

    return ''

if __name__ == '__main__':
    app.run_server(debug=True)
