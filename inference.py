import os
import numpy as np
import pandas as pd
from models import DeGrootModel, FriedkinModel, HegselmannKrauseModel
import dash
from dash import dcc, html
import dash_daq as daq
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from funcs import *
from models.degroot_inf import InfiniteDeGrootModel
from models.friedkin_inf import InfiniteFriedkinModel
from models.hegselmann_inf import InfiniteHegselmannKrauseModel


MAT = ''
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
            value='DeGroot'
            
        ),
        

        dcc.Graph(id='model-graph', className='graph'),

        html.Div(
            id = 'sl',
            children=[dcc.RangeSlider(
                id='n-slider',
                min=0,
                max=50,
                step=1,
                value=[0, 10],
                marks={i: str(i) for i in range(0, 51)}
                
            )], style={'display': 'block'}
        ),


        html.Div(
            id= 'ag-num',
            children=[
            html.Label('Number of Agents'),
            dcc.Input(id='agent-count-input', value='3', type='number', min=2),
            html.Button('Apply', id='apply-agent-count-btn')
        ], style={'display': 'block'}),
        
        html.Div(
            id='stubbornness-container',
            children=[
                html.Label('Stubbornness '),
                dcc.Input(id='stubbornness-input', value='0.1', type='text')
            ],
            style={'display': 'none'}
        ),
        
        html.Div(
            id='epsilon-container',
            children=[
                html.Label('Epsilon'),
                dcc.Input(id='epsilon-input', value='0.5', type='text')
            ],
            style={'display': 'none'}
        ),
        
        html.Div(
            id = 'in-st-g',
            children = [
            html.Label('Initial State Generation Method'),
            dcc.Dropdown(
                id='initial-state-method-dropdown',
                options=[
                    {'label': 'Equal Spacing', 'value': 'equal_spacing'},
                    {'label': 'Asymptotic Spacing', 'value': 'asymptotic_spacing'},
                    {'label': 'Random Spacing', 'value': 'random_spacing'}
                ],
                value='equal_spacing',
                style={'display': 'block'}
            ),
            html.Div(
                id='delta-cont',
                children=[
                    html.Label('Delta'),
                    dcc.Input(id='delta-input', value='0.1', type='text')
                ],
                style={'display': 'none'}
            ),
            html.Div(
                id='c-cont',
                children=[
                    html.Label('C (for Asymptotic Spacing)'),
                    dcc.Input(id='c-input', value='1.0', type='text')
                ],
                style={'display': 'none'}
            )
        ]),
        
        html.Div(
            id='in-st',
            children=[
                html.Label('Initial State (comma separated)'),
                dcc.Input(id='initial-state-input', value='0.6,0.4,0.7', type='text'),
                html.Button('Generate Initial State', id='generate-initial-state-btn'),
                html.Button('Save Initial State', id='save-initial-state-btn')
            ],
            style={'display': 'block'}
        ),

       
        html.Div(
            id='mat-container',
            children=[
                


                html.Label('Influence Matrix (comma separated rows)'),

                html.Div(children=[
                html.Label('Round'),
                daq.ToggleSwitch(
                    id='round_toggle',
                    value=False
                ),
                    
                ]),
                dcc.Textarea(
                    id='influence-matrix-input',
                    value='0.5,0.3,0.2\n0.2,0.5,0.3\n0.3,0.2,0.5',
                    style={'width': '100%', 'height': 100, 'display': 'block'}
                ),
                dcc.Textarea(
                    id='round-influence-matrix-input',
                    value='0.5,0.3,0.2\n0.2,0.5,0.3\n0.3,0.2,0.5',
                    style={'width': '100%', 'height': 100, 'display': 'none'}
                ),
                html.Button('Generate Influence Matrix', id='generate-matrix-btn', style={'display': 'block'} ),
                html.Button('Save Influence Matrix', id='save-matrix-btn')
            ],
            style={'display': 'block'}
        ),
       
        
        html.Div(
            id='inf-cont',
            children=[
                html.Label('Число шагов T'),
                dcc.Input(
                    id='T-input',
                    type='number',
                    value=10,
                    placeholder='Число шагов T'
                ),
                html.Br(),
                html.Label('Параметр K'),
                dcc.Input(
                    id='K-input',
                    type='number',
                    value=3,
                    placeholder='Параметр K'
                ),
                html.Br(),
                html.Label(id='E-input-label', children=['Порог epsilon'], style={'display': 'block'} ),
                dcc.Input(
                    id='E-input',
                    type='number',
                    value=0.2,
                    placeholder='Порог epsilon',
                    style={'display': 'block'}
                )
            ],
            style={'margin-top': '10px', 'display': 'none'}
        ),

        
        
        
        
        html.Div(id='save-output', style={'marginTop': 20})
    ]
)



@app.callback(
    [Output('influence-matrix-input', 'style'), 
     Output('round-influence-matrix-input', 'style') ],
    Input('round_toggle', 'value')
)
def update_output(value):
    if value:
        return {'width': '100%', 'height': 100, 'display': 'none'}, {'width': '100%', 'height': 100, 'display': 'block'}
    return {'width': '100%', 'height': 100, 'display': 'block'}, {'width': '100%', 'height': 100, 'display': 'none'}


@app.callback(
    Output('stubbornness-container', 'style'),
    [Input('model-dropdown', 'value')]
)
def toggle_stubbornness_input(selected_model):
    if selected_model == 'Friedkin' or selected_model == 'InfiniteFriedkin' :
        return {'display': 'block'}
    return {'display': 'none'}

@app.callback(
    Output('epsilon-container', 'style'),
    [Input('model-dropdown', 'value')]
)
def toggle_epsilon_input(selected_model):
    if selected_model == 'Hegselmann-Krause':
        return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    Output('inf-cont', 'style'),
    [Input('model-dropdown', 'value')]
)
def toggle_inf_input(selected_model):
    if 'Infinite' in selected_model:
        return {'display': 'block'}
    return {'display': 'none'}



@app.callback(
    Output('ag-num', 'style'),
    [Input('model-dropdown', 'value')]
)
def toggle_ag_num_input(selected_model):
    if 'Infinite' in selected_model:
        return {'display': 'none'}
    return {'display': 'block'}


@app.callback(
    Output('in-st-g', 'style'),
    [Input('model-dropdown', 'value')]
)
def toggle_ag_num_input(selected_model):
    if 'Infinite' in selected_model:
        return {'display': 'none'}
    return {'display': 'block'}



# @app.callback(
#     Output('sl', 'style'),
#     [Input('model-dropdown', 'value')]
# )
# def toggle_ag_num_input(selected_model):
#     if 'Infinite' in selected_model:
#         return {'display': 'none'}
#     return {'display': 'block'}



@app.callback(
    Output('generate-matrix-btn', 'style'),
    [Input('model-dropdown', 'value')]
)
def toggle_mat_input(selected_model):
    if 'Infinite' in selected_model:
        return {'display': 'none'}
    return {'display': 'block'}


@app.callback(
    Output('in-st', 'style'),
    [Input('model-dropdown', 'value')]
)
def toggle_mat_input(selected_model):
    if 'Infinite' in selected_model:
        return {'display': 'none'}
    return {'display': 'block'}


# @app.callback(
#     Output('delta-cont', 'style'),
#     [Input('initial-state-method-dropdown', 'value')]
# )
# def toggle_delta_input(selected_model):
#     if selected_model == 'equal_spacing':
#         return {'display': 'block'}
#     return {'display': 'none'}


@app.callback(
    [Output('c-cont', 'style'),
     Output('delta-cont', 'style')],
    [Input('initial-state-method-dropdown', 'value')]
)
def toggle_c_input(selected_model):
    if selected_model == 'asymptotic_spacing':
        return [{'display': 'block'}, {'display': 'block'}] 
    elif selected_model == 'equal_spacing':
        return [{'display': 'none'}, {'display': 'block'}] 
    return [{'display': 'none'}, {'display': 'block'}] 





@app.callback(
    Output('mat-container', 'style'),
    [Input('model-dropdown', 'value')]
)
def toggle_mat_input(selected_model):
    if selected_model == 'Hegselmann-Krause' or selected_model == 'InfiniteHegselmannKrause' :
        return {'display': 'none'}
    return {'display': 'block'}



@app.callback(
    [Output('E-input', 'style'),
     Output('E-input-label', 'style')],
    [Input('model-dropdown', 'value')]
)
def toggle_E_input(selected_model):
    if selected_model == 'InfiniteHegselmannKrause' :
        return {'display': 'block'}, {'display': 'block'}
    return {'display': 'none'}, {'display': 'none'}


@app.callback(
    [Output('initial-state-input', 'value'),
     Output('influence-matrix-input', 'value')],
    [Input('generate-initial-state-btn', 'n_clicks'), 
     Input('generate-matrix-btn', 'n_clicks'), 
     Input('apply-agent-count-btn', 'n_clicks'), 
     Input('initial-state-method-dropdown', 'value'), 
     Input('delta-input', 'value'),
     Input('c-input', 'value')],
    [State('agent-count-input', 'value'),
     State('initial-state-input', 'value'),
     State('influence-matrix-input', 'value')]
)
def update_inputs(generate_initial_state_n_clicks, generate_matrix_n_clicks, apply_agent_count_n_clicks, initial_config, delta, C, agent_count, initial_state_value, influence_matrix_value):
    ctx = dash.callback_context

    if not ctx.triggered:
        return initial_state_value, influence_matrix_value

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'apply-agent-count-btn':
        if initial_config == 'equal_spacing':
            initial_state = generate_equal_spacing(delta, agent_count)
        elif initial_config == 'asymptotic_spacing':
            initial_state = generate_asymptotic_spacing(delta, C, agent_count)
        elif initial_config == 'random_spacing':
            initial_state = generate_random_spacing(delta, agent_count)

        influence_matrix = generate_stochastic_matrix(agent_count)
        initial_state_str = ','.join(map(str, initial_state))
        influence_matrix_str = '\n'.join([','.join(map(str, row)) for row in influence_matrix])
        return initial_state_str, influence_matrix_str
    
  

    if trigger_id == 'generate-initial-state-btn':
        if initial_config == 'equal_spacing':
            initial_state = generate_equal_spacing(delta, agent_count)
        elif initial_config == 'asymptotic_spacing':
            initial_state = generate_asymptotic_spacing(delta, C, agent_count)
        elif initial_config == 'random_spacing':
            initial_state = generate_random_spacing(delta, agent_count)

        # initial_state = generate_random_initial_state(agent_count)
        initial_state_str = ','.join(map(str, initial_state))
        return initial_state_str, influence_matrix_value

    if trigger_id == 'generate-matrix-btn':
        influence_matrix = generate_stochastic_matrix(agent_count)
        influence_matrix_str = '\n'.join([','.join(map(str, row)) for row in influence_matrix])
        return initial_state_value, influence_matrix_str

    return initial_state_value, influence_matrix_value




   



@app.callback(
    [Output('model-graph', 'figure'),
     Output('influence-matrix-input', 'value', allow_duplicate=True),
     Output('round-influence-matrix-input', 'value')],
    [Input('model-dropdown', 'value'),
     Input('initial-state-input', 'value'),
     Input('influence-matrix-input', 'value'),
     Input('n-slider', 'value'),
     Input('stubbornness-input', 'value'),
     Input('epsilon-input', 'value'),
     Input('T-input', 'value'),
     Input('K-input', 'value'),
     Input('E-input', 'value'),
     Input('delta-input', 'value'),
     Input('c-input', 'value'),
     
     ]
     , prevent_initial_call=True

)
def update_graph(selected_model, initial_state_str, influence_matrix_str, n_range, stubbornness_str, epsilon_str, T, K, E, delta, C):


    if 'Infinite' not in selected_model:
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
        stubbornness = float(stubbornness_str)  # Пример жесткости агентов
        model = FriedkinModel(initial_state, influence_matrix, stubbornness)
    elif selected_model == 'Hegselmann-Krause':
        epsilon = float(epsilon_str)  # Пример параметра для модели Хегсельмана-Краузе
        model = HegselmannKrauseModel(initial_state, epsilon)

    if selected_model == 'InfiniteDeGroot':
        model = InfiniteDeGrootModel(float(delta), int(K), int(T))
        model.adjust_participants(T)

    elif selected_model == 'InfiniteFriedkin':
        stubbornness = float(stubbornness_str) 
        model = InfiniteFriedkinModel(float(delta), int(K), int(T))
        model.stubbornness = stubbornness
        model.adjust_participants(T)
    elif selected_model == 'InfiniteHegselmannKrause':
        model = InfiniteHegselmannKrauseModel(float(delta), int(K), int(T))
        model.epsilon = E
        model.adjust_participants(T)

    # Генерация состояний



    if 'Infinite' in selected_model:
        # states_up_to_n = model.generate_states_up_to_n(n_range[1])
        # time_steps = list(range(n_range[1] + 1))
        # states = np.array(states_up_to_n).T


        L1 = model.L1
        num_display_agents = 100
        start_agent = max((L1 - num_display_agents) // 2, 0)
        end_agent = start_agent + num_display_agents
        states_up_to_n = model.generate_states_up_to_n(n_range[1])
        states = np.array(states_up_to_n).T[start_agent:end_agent]
        time_steps = list(range(n_range[0], n_range[1] + 1))
        traces = []
        print(states.shape)
        for i in range(states.shape[0]): # Ограничение на отображение до 100 агентов
            traces.append(go.Scatter(
                x=time_steps,
                y=states[i],
                mode='lines+markers',
                name=f'Agent {i+1}'
            ))

    


    else:
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


    if 'Infinite' in selected_model:
        if selected_model == 'InfiniteHegselmannKrause':
            round_influence_matrix_ret =''
            influence_matrix_ret = ''
        else:
            # if model.influence_matrix.shape[0] > 20:
            #     round_influence_matrix_ret ='Матрица слишком большая, показать не получается, зато можно сохранить в csv 👇🏽'
            #     influence_matrix_ret = 'Матрица слишком большая, показать не получается, зато можно сохранить в csv 👇🏽'
            # else:
            mat = np.round(model.influence_matrix, 2)
            round_influence_matrix_ret ='\n'.join([','.join(map(str, row)) for row in mat]) 
            influence_matrix_ret = '\n'.join([','.join(map(str, row)) for row in model.influence_matrix])

            
    else:
        matrix = np.array(
            [list(map(float, row.split(','))) for row in influence_matrix_str.split('\n')]
        )
        mat = np.round(matrix, 2)
        round_influence_matrix_ret = '\n'.join([','.join(map(str, row)) for row in mat]) 
        influence_matrix_ret = influence_matrix_str
    # MAT = influence_matrix_ret
    return {
        'data': traces,
        'layout': go.Layout(
            title=f'Состояния модели {selected_model} от шага {n_range[0]} до {n_range[1]}',
            xaxis={'title': 'Шаги'},
            yaxis={'title': 'Состояние'},
            hovermode='closest'
        )
    }, influence_matrix_ret, round_influence_matrix_ret

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
        df_initial_state.to_csv('data/initial_state.csv', index=False)
        return 'Initial state saved to initial_state.csv'
    
    if trigger_id == 'save-matrix-btn':
        influence_matrix = np.array(
            [list(map(float, row.split(','))) for row in influence_matrix_value.split('\n')]
        )
        df_influence_matrix = pd.DataFrame(influence_matrix)
        df_influence_matrix.to_csv('data/influence_matrix.csv', index=False)
        return 'Influence matrix saved to influence_matrix.csv'

    return ''

if __name__ == '__main__':
    try:
        os.mkdir('data')
    except:
        pass
    app.run_server(debug=True)
