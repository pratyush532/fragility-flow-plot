
import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
import sklearn

def generate_dataframe(nb, ns, soil_type, prediction_type):
    # Load the models from pickle files
    with open('rf_Sd.pkl', 'rb') as file:
        loaded_rf_Sd_Matlab = pickle.load(file)

    with open('rf2_Sa.pkl', 'rb') as file:
        loaded_rf_Sa_Matlab = pickle.load(file)

    df = pd.read_csv(r'df_without_target.csv')
    df = df[['Nb', 'Ns', 'SOIL TYPE_I (HARD)', 'SOIL TYPE_II(MEDIUM)',
             'SOIL TYPE_III(SOFT)', 'Z', 'dy', 'Op']]

    if soil_type == 'Hard':
        df_trial = df.loc[(df['Nb'] == nb) & (df['Ns'] == ns) & (df['SOIL TYPE_I (HARD)'] == 1)]
    elif soil_type == 'Medium':
        df_trial = df.loc[(df['Nb'] == nb) & (df['Ns'] == ns) & (df['SOIL TYPE_II(MEDIUM)'] == 1)]
    elif soil_type == 'Soft':
        df_trial = df.loc[(df['Nb'] == nb) & (df['Ns'] == ns) & (df['SOIL TYPE_III(SOFT)'] == 1)]

    if prediction_type == 'Sd':
        predictions = loaded_rf_Sd_Matlab.predict(df_trial)
        df_trial['predictions'] = predictions
    elif prediction_type == 'Sa':
        predictions = loaded_rf_Sa_Matlab.predict(df_trial)
        df_trial['predictions'] = predictions

    return df_trial

def z_function(x, y, beta):
    return norm.cdf((1 / beta) * np.log(y / x))

def plot_fragility_flow(df_trial, x_vals, y_vals, beta, nb, ns, ncontours, 
                        height, width, prediction_type, toggle_state_ticks, soil_type):
    type_letter = prediction_type[-2]
    x = df_trial['predictions']
    y = df_trial['predictions']
    
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = z_function(X, Y, beta)

    custom_colorscale = ['blue', 'green','yellow', 'orange','red']
    n_contours = 9

    contour_plot = go.Figure(data=go.Contour(x=x_vals, y=y_vals, z=Z, contours_coloring='lines', 
                                             colorbar={
                                                            "title": "Contour legend",
                                                            "nticks":10,
                                                            "titleside": "right",
                                                            "ticks": "outside",
                                                            "tickcolor": "black",
                                                            "tickfont": {"size": 10, "color": "black"}
                                                        },
                                             colorscale=custom_colorscale, contours={"showlabels": True,
                         "labelfont": {"size": 12, "color": 'green'}}))
    
    if type_letter=='d':
        contour_plot.update_traces(hovertemplate='Sd_bar: %{x}<br>Sd: %{y}<br>Fragility: %{z}',name="")
    else:
        contour_plot.update_traces(hovertemplate='Sa_bar: %{x}<br>Sa: %{y}<br>Fragility: %{z}',name="")
        
    if toggle_state_ticks:
        contour_plot.add_trace(go.Scatter(x=x, y = [0]*len(x), mode='markers', name="S" + "\u0304"+'<sub>'+type_letter+'</sub>) (m)'))
        contour_plot.add_trace(go.Scatter(x=[0]*len(y), y = y, mode='markers', name=r'S<sub>'+type_letter+'</sub>, m)'))
        contour_plot.update_layout(
            title=dict(text=f'Fragility Flow Plot for {nb}B-{ns}S : {prediction_type}|  \u03B2 = {beta}  | Soil Type : {soil_type}', font=dict(size=18)),
            xaxis_title=prediction_type[:-4] + ' Threshold (' + "S" + "\u0304"+'<sub>'+type_letter+'</sub>) (m)',
            yaxis_title=prediction_type + ' (' + r'S<sub>'+type_letter+'</sub>, m)',
            xaxis=dict(showline=True, showgrid=True, tickvals = x),
            yaxis=dict(showline=True, showgrid=True, tickvals = y),
            height= height,
            width=width,
        )
    else:
        contour_plot.update_layout(
            title=dict(text=f'Fragility Flow Plot for {nb}B-{ns}S : {prediction_type} |  \u03B2 = {beta} | Soil Type : {soil_type}', font=dict(size=18)),
            xaxis_title=prediction_type[:-4] + ' Threshold (' + "S" + "\u0304"+'<sub>'+type_letter+'</sub>) (m)',
            yaxis_title=prediction_type + ' (' + r'S<sub>'+type_letter+'</sub>, m)',
            xaxis=dict(showline=True, showgrid=True),
            yaxis=dict(showline=True, showgrid=True),
            height= height,
            width=width,
        )
        
    contour_plot.update_traces(ncontours=ncontours, selector=dict(type='contour'))

    st.plotly_chart(contour_plot)

def main():
    st.set_page_config(layout="wide")

    # Set the title of the app
    st.title('Interactive Plot to Generate Fragility Flow ')

    st.sidebar.markdown("# Prediction Type")

    # Input parameters
    # Add a radio button for selecting prediction type
    prediction_type = st.sidebar.radio("Select Prediction Type", ['Spectral Displacement (Sd)', 'Spectral Acceleration (Sa)'])
    st.sidebar.markdown("# Input Parameters")
    nb = st.sidebar.slider("Number of stories (Nb)", min_value=1, max_value=6, value=6)
    ns = st.sidebar.slider("Number of spans (Ns)", min_value=2, max_value=6, value=6)
    soil_type = st.sidebar.radio("Select Soil Type", ['Hard', 'Medium', 'Soft'])
    
    

    df_trial = generate_dataframe(nb, ns, soil_type, prediction_type[-3:-1])
    #st.write(df_trial)
    #st.write(len(df_trial['predictions']))
    
    original_array = np.sort(np.array(df_trial['predictions']))
    #st.write(original_array)

    # Initialize an empty list to store the result
    result_array = []
    # Iterate through pairs of consecutive numbers
    for i in range(len(original_array) - 1):
        # Add the current number to the result array
        result_array.append(original_array[i])
        # Calculate the spacing between the current and next numbers
        spacing = (original_array[i + 1] - original_array[i]) / 5
        # Add 4 equally spaced numbers between the current and next numbers
        for j in range(1, 5):
            result_array.append(original_array[i] + j * spacing)
    # Add the last number of the original array to the result
    result_array.append(original_array[-1])

    # Convert the result to a NumPy array if needed
    result_array = np.array(result_array)
    x_vals, y_vals = result_array, result_array

    #x_vals = np.linspace(np.min(df_trial['predictions']), np.max(df_trial['predictions']), 100)
    #y_vals = np.linspace(np.min(df_trial['predictions']), np.max(df_trial['predictions']), 100)

    beta = st.sidebar.slider("Select Beta (0.1 to 1)", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    ncontours = st.sidebar.slider("Density of Contour Plot", min_value=20, max_value=70, value=30, step=5)
    
    toggle_state_ticks = st.sidebar.checkbox("Show Data Points", value=False)
    

    st.sidebar.markdown("# Graph Dimensions")

    height = st.sidebar.slider("Select Y Dimension", min_value=500, max_value=1500, value=600, step=10)
    width = st.sidebar.slider("Select X Dimension", min_value=500, max_value=1500, value=1000, step=10)
    plot_fragility_flow(df_trial, x_vals, y_vals, beta, nb, ns, ncontours, height, width, prediction_type, toggle_state_ticks, soil_type)
    
#     print(f"Streamlit version: {st.__version__}")
#     print(f"Pandas version: {pd.__version__}")
#     print(f"Numpy version: {np.__version__}")
#     print(f"Scipy version: {sklearn.__version__}")
    

if __name__ == "__main__":
    main()
