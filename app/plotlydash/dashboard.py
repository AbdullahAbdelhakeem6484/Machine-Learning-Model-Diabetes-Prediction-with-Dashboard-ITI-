"""Instantiate a Dash app."""
import numpy as np
import pandas as pd
import dash
import dash_table
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
import pickle



#___________________________________________________________
"""Plotly Dash HTML layout override."""

html_layout = """
<!doctype html>
<html lang="en" class="h-100">
<head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous">
    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css" integrity="sha384-giJF6kkoqNQ00vy+HMDP7azOuL0xtbfIcaT9wjKHr8RbDVddVHyTfAAsrekwKmP1" crossorigin="anonymous">
    <!-- Google Fonts -->
    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap"/>
    <!-- Custom CSS for Dash -->
    <link rel="stylesheet" href="/static/dashstyles.css">
    <!-- Custom CSS -->
    <link rel="stylesheet" href="/static/style.css">

    <!-- bootstrap Bundle with Popper -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/js/bootstrap.bundle.min.js" async integrity="sha384-ygbV9kiqUc6oa4msXn9868pTtWMgiQaeYH7/t7LECLbyPA2x65Kgf80OJFdroafW" crossorigin="anonymous"></script>
    <title>Data Visualizations</title>
</head>
<body class="d-flex flex-column h-100 body">
<nav class="navbar navbar-expand-lg navbar-light bg-white shadow">
    <div class="container">
        <a class="navbar-brand" href="#"><span class="fa fa-medkit"></span> Machine Learning Diabetes Project</a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                <li class="nav-item">
                    <a class="nav-link " href="/">Home</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link " href="/diagnose">Diagnosis</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link active" href="#">Data Visualizations</a>
                </li>
                
            </ul>
        </div>
    </div>
</nav>
<main role="main" class="flex-shrink-0">
    <div class="container shadow rounded fadeIn p-5 my-5 bg-white" id="content">

    <h2> Data Visualization </h2>
    <br>
            {%app_entry%}
    </div>
</main>
<footer class="footer mt-auto py-3 text-white">
    <div class="container">
        <div class="d-flex justify-content-between">
            <div>
                <span class="text-white small">Abdullah W Sebaie @ ITI</span>
            </div>
            <div class>
                <a class="text-white text-decoration-none" href="https://github.com/">View the source on Github <i class="fa fa-github"></i></a>
            </div>
        </div>
    </div>
                {%config%}
                {%scripts%}
                {%renderer%}
</footer>
</body>
</html>
"""



#___________________________________________________________
# rf_model = pickle.load("rf_model.pkl")

with open('app/data/rf_model.pkl', 'rb') as pickle_file:
    rf_model = pickle.load(pickle_file)



def init_dashboard(server):
    """Create a Plotly Dash dashboard."""
    dash_app = dash.Dash(
        server=server,
        routes_pathname_prefix='/dashapp/'
    )

    # Load DataFrame
    df  = pd.read_csv("app/data/diabetes_clean.csv")


    # Create confusion matrix
    feature_heatmap_fig = px.imshow(df.corr(), color_continuous_scale=px.colors.sequential.Plasma)

    # Create feature importance bar chart
    feature_importance = pd.Series(rf_model.feature_importances_*100,
                                   index=df.columns[:-1]).sort_values(ascending=False)
    feat_importance_fig = px.bar(feature_importance,
                                 labels={'value': 'Importance', 'index': 'Features'},color_discrete_sequence =['brown']
)
    feat_importance_fig.layout.update(showlegend=False)

    # Create age/feature scatter plot
    x_graph = df[df['class'] == True]['Age'].value_counts().index
    y_graph = df[df['class'] == True]['Age'].value_counts().values
    age_scatter_fig = go.Figure()
    age_scatter_fig.add_trace(go.Scatter(x=x_graph, y=y_graph,
                                         mode='markers',
                                         name='Diabetes',
                                         marker={"size": 12, "color": 'orange'}))
    x_graphb = df[df['Polyuria'] == True]['Age'].value_counts().index
    y_graphb = df[df['Polyuria'] == True]['Age'].value_counts().values
    age_scatter_fig.add_trace(go.Scatter(x=x_graphb, y=y_graphb,
                                         mode='markers',
                                         name='Polyuria',
                                         marker={"size": 12, "color": 'MediumPurple'}))
    age_scatter_fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Number of Patients",
        legend_title="Condition",
        )

    # Custom HTML layout
    dash_app.index_string = html_layout

    # Create Layout
    dash_app.layout = html.Div(
        children=[
            html.Br(),
            html.H5(children="Feature Heatmap", className="lh-1"),
            dcc.Graph(figure=feature_heatmap_fig),
            html.Br(),
            html.H5(children="Machine Learning Feature Importance", className="lh-1"),
            dcc.Graph(figure=feat_importance_fig),
            html.Br(),
            html.H5(children="Age Distribution", className="lh-1"),
            dcc.Graph(figure=age_scatter_fig),
            html.Br(),
            html.H5(children="Dataset Browser", className="lh-1"),
            html.Br(),
            html.Br(),
            create_data_table(df),
        ],
        id='dash-container'
    )
    return dash_app.server


def create_data_table(df):
    """Create Dash datatable from Pandas DataFrame."""
    table = dash_table.DataTable(
        id='database-table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'scroll'},
        sort_action="native",
        sort_mode='native',
        page_size=10
    )
    return table
