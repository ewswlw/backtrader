import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import pandas as pd
import os

# Set up logging
logger = logging.getLogger(__name__)

def create_spread_plots(df: pd.DataFrame, output_path: str, title: str = "Data Series Over Time"):
    """
    Create interactive plots for multiple data series and save as HTML.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data series to plot
        output_path (str): Path to save the HTML file
        title (str): Title for the plot
        
    Returns:
        str: Path to the saved HTML file
    """
    try:
        # Calculate layout dimensions
        n_series = len(df.columns)
        n_rows = math.ceil(n_series / 3)
        n_cols = min(3, n_series)
        vertical_spacing = min(0.08, 1.0 / (n_rows + 1))
        
        # Create subplot grid
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=df.columns,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=0.05
        )
        
        # Add each series to a subplot
        for idx, column in enumerate(df.columns):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[column],
                    name=column,
                    line=dict(width=1),
                    showlegend=False,
                    hovertemplate=
                    "<b>%{x}</b><br>" +
                    column + ": %{y:.2f}<br>" +
                    "<extra></extra>"
                ),
                row=row,
                col=col
            )
            
            # Update axes labels
            fig.update_xaxes(
                title_text="Date",
                row=row,
                col=col,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)',
                tickangle=45,
                tickformat='%Y-%m-%d'
            )
            fig.update_yaxes(
                title_text=column,
                row=row,
                col=col,
                showgrid=True,
                gridcolor='rgba(128, 128, 128, 0.2)'
            )

        # Update layout
        fig.update_layout(
            template='plotly_dark',
            showlegend=False,
            height=250 * n_rows,
            title={
                'text': title,
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            paper_bgcolor='rgb(30, 30, 30)',
            plot_bgcolor='rgb(30, 30, 30)',
            margin=dict(t=80, l=50, r=50, b=50),
            font=dict(
                family="Arial",
                size=10,
                color="white"
            ),
            autosize=True
        )
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to HTML file
        fig.write_html(
            output_path,
            include_plotlyjs=True,
            full_html=True,
            config={
                'responsive': True,
                'displayModeBar': True,
                'scrollZoom': True,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']
            }
        )
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating plots: {str(e)}")
        raise
