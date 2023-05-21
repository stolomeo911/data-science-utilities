import plotly.graph_objs as go


def plot_chart_multiple_columns(df, x_column, y_columns, y_format, x_label_mode, x_tick_format, xtick):
    traces = []
    for col in y_columns: # Skip the first column ('x')
        trace = go.Scatter(
            x=df[x_column],
            y=df[col],
            mode='lines+markers+text',
            name=col,
            line=dict(
                width=3,
            ),
            text=[f"{num:{y_format}}" for num in df[col]],
            hoverinfo='text',
        )
        traces.append(trace)

    # Define the layout for the line chart
    layout = go.Layout(
        title='Line Chart Example',
        xaxis=dict(
            title='Time',
            tickformat=x_tick_format,
            ticklabelmode=x_label_mode,
            dtick=xtick
        ),
        yaxis=dict(
            title='Value',
            tickformat=",.0%"
        ),
        hovermode='closest',
    )

    # Create the figure for the line chart
    fig = go.Figure(data=traces, layout=layout)
    fig.update_traces(textposition='top center')

    # Set the resolution and quality of the exported image
    fig.update_layout(
        width=1000,
        height=600,
        margin=dict(l=40, r=20, t=30, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    fig.show()