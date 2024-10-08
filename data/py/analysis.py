import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_custom_hover_text(x, y, name, additional_metric=None):
    """Create custom hover text for the plot with detailed metrics."""
    
    if additional_metric is not None:
        return [
            f"{name}<br>Count: {y_val}<br>X Metric: {x_val}<br>{additional_metric} occurred."
            for x_val, y_val in zip(x, y)
        ]
    else:
        return [
            f"{name}<br>Count: {y_val}<br>X Metric: {x_val}<br>No additional metrics."
            for x_val, y_val in zip(x, y)


        ]

hover_text_call_volume = create_custom_hover_text(x, y1, "Call Volume", additional_metric="There were X abandoned calls.")
hover_text_aht = create_custom_hover_text(x, y2, "Avg Handle Time")
hover_text_csat = create_custom_hover_text(x, y3, "Customer Satisfaction Score", additional_metric="Satisfaction level noted.")
hover_text_fcr = create_custom_hover_text(x, y4, "First Call Resolution Rate")

def create_kpi_visualization(x, y1, y2, y3, y4, call_volume_mean, aht_mean, csat_mean, fcr_mean):
    """Create subplots for KPI visualization."""
    
    # Check input data types for security.
    if not (isinstance(x, (list, np.ndarray)) and isinstance(y1, (list, np.ndarray)) and 
            isinstance(y2, (list, np.ndarray)) and isinstance(y3, (list, np.ndarray)) and 
            isinstance(y4, (list, np.ndarray))):
        raise ValueError("Inputs must be lists or numpy arrays.")

    # Create subplots for visualizing KPI metrics.
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Call Volume", 
            "Avg Handle Time", 
            "Customer Satisfaction Score", 
            "First Call Resolution Rate"
        )
    )

    # Add Call Volume bar plot.
    fig.add_trace(go.Bar(x=x, y=y1, name="Call Volume", hovertext=create_custom_hover_text(x, y1, "Call Volume")), row=1, col=1)

    # Add Avg Handle Time line plot.
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name="Avg Handle Time", 
                             hovertext=create_custom_hover_text(x, y2, "Avg Handle Time")), row=1, col=2)

    # Add Customer Satisfaction Score scatter plot.
    fig.add_trace(go.Scatter(x=x, y=y3, mode='markers', name="CSAT", 
                             hovertext=create_custom_hover_text(x, y3, "CSAT")), row=2, col=1)

    # Add First Call Resolution Rate box plot.
    fig.add_trace(go.Box(y=y4, name="FCR", hovertext=create_custom_hover_text(x, y4, "FCR")), row=2, col=2)

    # Add mean lines for each metric.
    fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[call_volume_mean, call_volume_mean], mode='lines', 
                             name='Call Volume Mean', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[aht_mean, aht_mean], mode='lines', 
                             name='AHT Mean', line=dict(color='green', dash='dash')), row=1, col=2)
    fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[csat_mean, csat_mean], mode='lines', 
                             name='CSAT Mean', line=dict(color='blue', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[fcr_mean, fcr_mean], mode='lines', 
                             name='FCR Mean', line=dict(color='purple', dash='dash')), row=2, col=2)

    # Update layout with theme and colors.
    fig.update_layout(
        title_text="KPI Metrics",
        height=600,
        width=1300,
        showlegend=True,
        template='plotly_dark',
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
    )
    
    # Make axes clear and set grid colors.
    fig.update_xaxes(showgrid=False, color='white')
    fig.update_yaxes(showgrid=False, color='white')

    # Show the figure.
    fig.show()

create_kpi_visualization(x, y1, y2, y3, y4, call_volume_mean, aht_mean, csat_mean, fcr_mean)

#======================================================================================================================#

import os
import plotly.io as pio

def export_kpi_visualization_as_html(fig, file_name, directory, width=800, height=600):
    """Export the KPI visualization as an HTML file."""
    
    # Create the directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Function to get the next file number by checking existing files
    def get_next_file_number(base_name):
        i = 1
        while True:
            new_file_name = f"{base_name}_{i:02d}.html"
            full_path = os.path.join(directory, new_file_name)
            if not os.path.exists(full_path):
                return full_path
            i += 1

    # Get the next available HTML file name with path
    next_file_path = get_next_file_number(file_name)

    # Set the plotly_dark theme and dimensions, ensuring black background
    fig.update_layout(
        template='plotly_dark',
        width=width,
        height=height,
        plot_bgcolor='black',  # Set plot background to black
        paper_bgcolor='black'   # Set paper background to black
    )

    # Export the figure to HTML
    pio.write_html(fig, file=next_file_path)

    print(f"File saved as: {next_file_path}")

# Example usage
file_name = "dashboard_vis"
directory = os.path.join("..", "..", "static", "graphs")

export_kpi_visualization_as_html(fig, file_name=file_name, directory=directory, width=1300, height=800)

#======================================================================================================================#

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_custom_hover_text(x, y, name, additional_metric=None):
    """Create custom hover text for the plot with detailed metrics."""
    
    if additional_metric is not None:
        return [
            f"{name}<br>Count: {y_val}<br>X Metric: {x_val}<br>{additional_metric} occurred."
            for x_val, y_val in zip(x, y)
        ]
    else:
        return [
            f"{name}<br>Count: {y_val}<br>X Metric: {x_val}<br>No additional metrics."
            for x_val, y_val in zip(x, y)
        ]

def create_kpi_visualization(x, y1, y2, y3, y4, call_volume_mean, aht_mean, csat_mean, fcr_mean, file_path):
    """Create subplots for KPI visualization."""
    
    # Check input data types for security.
    if not (isinstance(x, (list, np.ndarray)) and isinstance(y1, (list, np.ndarray)) and 
            isinstance(y2, (list, np.ndarray)) and isinstance(y3, (list, np.ndarray)) and 
            isinstance(y4, (list, np.ndarray))):
        raise ValueError("Inputs must be lists or numpy arrays.")

    # Create subplots for visualizing KPI metrics.
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}], 
               [{'type': 'scatter'}, {'type': 'bar'}]],  # 2D scatter and horizontal bar in the second row
        subplot_titles=(
            "3D Scatter Plot of Call Volume", 
            "2D Scatter Plot of Avg Handle Time", 
            "Line Plot of Customer Satisfaction Score", 
            "Trends in First Call Resolution Rate (Horizontal Bar)"
        )
    )

    # Add 3D Scatter plot for Call Volume.
    fig.add_trace(go.Scatter3d(x=x, y=y1, z=np.zeros_like(y1), mode='markers', name="3D Scatter Call Volume",
                                 hovertext=create_custom_hover_text(x, y1, "Call Volume")),
                  row=1, col=1)

    # Add 2D Scatter plot for Avg Handle Time.
    fig.add_trace(go.Scatter(x=x, y=y2, mode='markers', name="Avg Handle Time",
                             hovertext=create_custom_hover_text(x, y2, "Avg Handle Time")),
                  row=1, col=2)

    # Add Line plot for Customer Satisfaction Score.
    fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name="CSAT Line Plot",
                             hovertext=create_custom_hover_text(x, y3, "Customer Satisfaction")),
                  row=2, col=1)

    # Add Horizontal Bar plot for First Call Resolution Rate.
    fig.add_trace(go.Bar(x=y4, y=x, orientation='h', name="FCR", 
                         hovertext=create_custom_hover_text(x, y4, "FCR")),
                  row=2, col=2)

    # Update layout with theme and colors.
    fig.update_layout(
        title_text="KPI Metrics",
        height=800,
        width=1300,
        showlegend=True,
        template='plotly_dark',
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
    )
    
    # Make axes clear and set grid colors.
    fig.update_xaxes(showgrid=False, color='white')
    fig.update_yaxes(showgrid=False, color='white')

    pio.write_html(fig, file=file_path)

    # Show the figure.
    fig.show()

# Sample data
x = np.arange(1, 11)
y1 = np.random.randint(50, 100, size=10)  # Call Volume
y2 = np.random.uniform(1, 5, size=10)  # Avg Handle Time
y3 = np.random.uniform(0, 100, size=10)  # Customer Satisfaction Score
y4 = np.random.uniform(0, 100, size=10)  # First Call Resolution Rate

# Sample mean values
call_volume_mean = np.mean(y1)
aht_mean = np.mean(y2)
csat_mean = np.mean(y3)
fcr_mean = np.mean(y4)

file_name = "test_graph.html"
file_path = os.path.join("..", "..", "static", "graphs", file_name)
create_kpi_visualization(x, y1, y2, y3, y4, call_volume_mean, aht_mean, csat_mean, fcr_mean, file_path)

#======================================================================================================================#

import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_monthly_call_volume_timeline(dates, call_volume):
    """Create a timeline plot for call volume data month on month."""
    
    # Convert inputs to Pandas Series if they are lists or numpy arrays for consistency.
    if isinstance(dates, list) or isinstance(dates, np.ndarray):
        dates = pd.Series(dates)
    if isinstance(call_volume, list) or isinstance(call_volume, np.ndarray):
        call_volume = pd.Series(call_volume)

    # Check input data types for security.
    if not (isinstance(dates, pd.Series) and isinstance(call_volume, pd.Series)):
        raise ValueError("Inputs must be Pandas Series.")

    # Create a DataFrame for easier handling.
    df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Call Volume': call_volume})

    # Sort by date
    df.sort_values('Date', inplace=True)

    # Create the timeline plot
    fig = go.Figure()

    # Add line plot for call volume
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Call Volume'],
        mode='lines+markers',
        name='Call Volume',
        hoverinfo='text',
        text=[f"Date: {date.strftime('%Y-%m')}<br>Volume: {volume}" for date, volume in zip(df['Date'], df['Call Volume'])]
    ))

    # Update layout
    fig.update_layout(
        title='Monthly Call Volume Timeline',
        xaxis_title='Date',
        yaxis_title='Call Volume',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True),
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        title_font=dict(size=20, color='white'),
        xaxis_title_font=dict(size=14, color='white'),
        yaxis_title_font=dict(size=14, color='white'),
        legend=dict(font=dict(color='white')),
        height=600,
        width=1300,
    )

    # Update x-axis ticks for better readability
    fig.update_xaxes(tickangle=-45, tickformat='%Y-%m')

    # Show the figure
    fig.show()

# Example data
dates = pd.date_range(start='2023-01-01', periods=12, freq='M')  # Monthly data for one year
call_volume = [100, 150, 120, 130, 160, 170, 190, 180, 200, 210, 230, 240]  # Example call volume data

create_monthly_call_volume_timeline(dates, call_volume)

#======================================================================================================================#

import plotly.express as px
import pandas as pd

def create_parallel_categories_plot(data, dimensions, color_column=None):
    """
    Create a simple parallel categories plot.
    
    :param data: Pandas DataFrame containing the categorical data.
    :param dimensions: List of column names to be used as dimensions in the plot.
    :param color_column: Column name for coloring the paths (optional).
    :return: A Plotly figure object.
    """
    # Create the plot using Plotly Express
    fig = px.parallel_categories(
        data_frame=data, 
        dimensions=dimensions, 
        color=color_column,
        color_continuous_scale=px.colors.sequential.Inferno  # Color scale for the plot
    )

    # Update layout for better visibility
    fig.update_layout(
        title="Parallel Categories Plot",
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white')
    )
    
    # Show the figure
    fig.show()

# Example data
data = pd.DataFrame({
    'Product Category': ['A', 'B', 'A', 'C', 'A', 'B', 'C', 'A'],
    'Region': ['North', 'South', 'North', 'East', 'West', 'West', 'East', 'South'],
    'Sales Channel': ['Online', 'Offline', 'Offline', 'Online', 'Offline', 'Online', 'Offline', 'Online'],
    'Profitability': ['High', 'Low', 'High', 'Medium', 'Medium', 'Low', 'High', 'Medium']
})

# Define the dimensions for the parallel categories plot
dimensions = ['Product Category', 'Region', 'Sales Channel', 'Profitability']

# Create the parallel categories plot
create_parallel_categories_plot(data, dimensions, color_column='Profitability')

#======================================================================================================================#