import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Seiteneinstellungen
st.set_page_config(
    page_title="Landmark Analysis Dashboard",
    page_icon="🎯",
    layout="wide"
)

# Titel und Einführung
st.title("🎯 Landmark Detection Analysis Dashboard")
st.markdown("""
            Dashboard shows the analysis of landmark detections and their accuracy.
""")

# Daten einlesen
@st.cache_data
def load_data():
    summary_df = pd.read_csv('/home/juval.gutknecht/Projects/Data/results/inference_results_aaa/report/tableau_landmark_summary.csv')
    details_df = pd.read_csv('/home/juval.gutknecht/Projects/Data/results/inference_results_aaa/report/tableau_detailed_errors.csv')
    return summary_df, details_df

summary_df, details_df = load_data()

# Dashboard Layout mit Spalten
tab1, tab2 = st.tabs(["📊 Summary", "🔍 Detailed Error analysis"])

with tab1:
    col1, col2 = st.columns(2)

    # First colon: Detection Rate und Inlier Rate
    with col1:
        st.subheader("Detection and Inlier Rates")
        
        # Balkendiagramm für Detection und Inlier Rates
        fig_rates = go.Figure(data=[
            go.Bar(name='Detection Rate', x=summary_df['Landmark'], y=summary_df['Detection Rate (%)'],
                marker_color='royalblue'),
            go.Bar(name='Inlier Rate (5mm)', x=summary_df['Landmark'], y=summary_df['Inlier Rate (5mm) (%)'],
                marker_color='lightgreen')
        ])
        
        fig_rates.update_layout(
            barmode='group',
            title='Detection and inlier Rates per Landmark',
            xaxis_tickangle=-45,
            height=500
        )
        st.plotly_chart(fig_rates, use_container_width=True)

    # second colon: Error Analysis
    with col2:
        st.subheader("Error Analysis")
        
        error_columns = ['Mean Error (mm)', 'Median Error (mm)', 'Std Error (mm)']
        error_df = summary_df[error_columns].reset_index()

        fig_error = px.box(summary_df, x='Landmark', y=['Mean Error (mm)', 'Median Error (mm)', 'Std Error (mm)'],
                        title='Error Distribution per Landmark')
        fig_error.update_layout(
            xaxis_tickangle=-45,
            height=500,
            yaxis_title='Error (mm)'  # Added y-axis description
        )
        st.plotly_chart(fig_error, use_container_width=True)


    # Key metrics in einer Reihe
    st.subheader("📊 Key metrics")
    cols = st.columns(4)

    # Key Metrics
    with cols[0]:
        avg_detection = summary_df['Detection Rate (%)'].mean()
        st.metric("Average Detection Rate", f"{avg_detection:.2f}%")

    with cols[1]:
        avg_inlier = summary_df['Inlier Rate (5mm) (%)'].mean()
        st.metric("Average Inlier Rate", f"{avg_inlier:.2f}%")

    with cols[2]:
        avg_mean_error = summary_df['Mean Error (mm)'].mean()
        st.metric("Average Mean Error", f"{avg_mean_error:.2f}mm")

    with cols[3]:
        avg_median_error = summary_df['Median Error (mm)'].mean()
        st.metric("Average Median Error", f"{avg_median_error:.2f}mm")

with tab2:
    st.subheader("Detailed error analysis per case")

    selected_image = st.selectbox(
        "Choose an image:",
        options=sorted(details_df['image'].unique())
    )

    col1, col2 = st.columns(2)

    with col1:
        image_data = details_df[details_df['image'] == selected_image]

        fig_error_magnitude = px.bar(
            image_data,
            x='landmark',
            y='error_magnitude',
            title=f'Error Magnitude per Landmark for {selected_image}',
            color='error_magnitude',
            color_continuous_scale='RdYlBu_r'
        )
        fig_error_magnitude.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_error_magnitude, use_container_width=True)

    with col2: 
        # 3D scatter plot
        fig_3d = px.scatter_3d(
            image_data,
            x='x_error',
            y='y_error',
            z='z_error',
            color='error_magnitude',
            hover_name='landmark',
            title=f'3D Error Distribution for {selected_image}'
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    
    st.subheader(f'Error details for {selected_image}')

# Formatted table with the most important colons
    display_cols = ['landmark', 'error_magnitude', 'x_error', 'y_error', 'z_error']
    formatted_data = image_data[display_cols].round(3)
    st.dataframe(
        formatted_data.style.background_gradient(subset=['error_magnitude'], cmap='RdYlBu_r'),
        height=400
    )

    # Error stats per axis
    st.subheader("Error stats per axis")
    error_stats = pd.DataFrame({
        'Statistics': ['Mean Error', 'Median Error', 'Max Error', 'Min Error'],
        'X-Axis': [
            f"{image_data['x_error'].mean():.2f}mm",
            f"{image_data['x_error'].median():.2f}mm",
            f"{image_data['x_error'].max():.2f}mm",
            f"{image_data['x_error'].min():.2f}mm"
        ],
        'Y-Axis': [
            f"{image_data['y_error'].mean():.2f}mm",
            f"{image_data['y_error'].median():.2f}mm",
            f"{image_data['y_error'].max():.2f}mm",
            f"{image_data['y_error'].min():.2f}mm"
        ],
        'Z-Axis': [
            f"{image_data['z_error'].mean():.2f}mm",
            f"{image_data['z_error'].median():.2f}mm",
            f"{image_data['z_error'].max():.2f}mm",
            f"{image_data['z_error'].min():.2f}mm"
        ]
    })
    st.table(error_stats)

# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit and Plotly")