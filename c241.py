import streamlit as st
import pandas as pd
import numpy as np
from joblib import dump, load
import os
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page title
st.title('Vehicle Price Prediction App')
st.write('Enter the details of the vehicle to predict its price')

# Define model paths and functions in a dictionary for better organization
MODEL_CONFIG = {
    'Linear Regression': {
        'model_path': 'LR_model.pkl',
        'module_name': 'c24_lr_func'
    },
    'Random Forest': {
        'model_path': 'RF_model.pkl',
        'module_name': 'c24_rf_func'
    },
    'Neural Network': {
        'model_path': 'NN_model.pkl',
        'module_name': 'c24_nn_func'
    }
}

# Function to load models dynamically
@st.cache_resource
def load_model(model_path):
    try:
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            return load(model_path)
        else:
            logger.error(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        return None

# Function to dynamically import preprocessing modules
def load_preprocessing_module(module_name):
    try:
        module = __import__(module_name, fromlist=['target_encoder', 'min_max_scaler_X', 'min_max_scaler_y'])
        return module
    except ImportError as e:
        logger.error(f"Error importing module {module_name}: {e}")
        st.error(f"Could not import preprocessing module {module_name}. Please check if it exists.")
        return None

# Define target encodings for 'make'
target_encodings = {
    'make': {
        'RENAULT': 4.644697986577181, 'BMW': 17.124458092485458, 'MAHINDRA': 7.320376827896513, 
        'HYUNDAI': 5.43734532955924, 'MARUTI': 4.683959677338039, 'TATA': 4.36807817158931, 
        'VOLKSWAGEN': 5.514684643510054, 'DATSUN': 3.1329869964257218, 'JEEP': 13.36133609672232, 
        'CHEVROLET': 2.601773584996439, 'TOYOTA': 10.442392601431976, 'AUDI': 17.17720651369338, 
        'KIA': 11.202214199232113, 'FORD': 5.887344249201276, 'NISSAN': 4.642875594838196, 
        'MERCEDES-BENZ': 17.02843846153785, 'HONDA': 5.949516721044046, 'SKODA': 7.405949477349784, 
        'MITSUBISHI': 8.189708322101406, 'FIAT': 3.156605283008319, 'PORSCHE': 12.667214057653903, 
        'MINI': 11.269942955752917, 'LAND': 17.1647135220591, 'VOLVO': 14.476712493735432, 
        'JAGUAR': 19.89061925255433, 'OPELCORSA': 5.786361314199528, 'ISUZU': 8.882671527148922, 
        'MG': 10.684943689450199, 'FORCE': 6.417094479957295, 'DC': 7.326492376657015, 
        'LEXUS': 9.636662950240016, 'MASERATI': 8.431616938965764, 'LAMBORGHINI': 8.780799198898078, 
        'BENTLEY': 5.707329016039025, 'AMBASSADOR': 5.907187265137702
    }
}

# Define default values
DEFAULT_VALUES = {
    'km_driven': 20000,
    'mileage': 20,
    'engine': 1500,
    'max_power': 100,
    'year': 2018,
    'make': 'MARUTI',
    'individual': 1,
    'vehicle_type': 1,
    'fuel_type': 4,
    'seat': 5
}

# Sidebar for model selection
with st.sidebar:
    st.header("Model Settings")
    selected_model = st.selectbox(
        'Select Prediction Model',
        options=list(MODEL_CONFIG.keys()),
        index=0  # Default to the first option
    )
    
    st.info(f"Using {selected_model} for prediction")
    
    # Check if model exists
    model_path = MODEL_CONFIG[selected_model]['model_path']
    if not os.path.exists(model_path):
        st.warning(f"Model file {model_path} not found! Using demo mode.")

# Create input fields in a more organized layout
st.subheader("Vehicle Details")
col1, col2 = st.columns(2)

with col1:
    km_driven = st.slider("KM Driven", 1000, 100000, DEFAULT_VALUES['km_driven'], step=1000)
    mileage = st.slider("Mileage", 5, 120, DEFAULT_VALUES['mileage'], step=1)
    engine = st.slider("Engine Power (CC)", 500, 5000, DEFAULT_VALUES['engine'], step=100)
    max_power = st.slider("Max Power (HP)", 60, 500, DEFAULT_VALUES['max_power'], step=10)
    year = st.slider("Year", 2010, 2023, DEFAULT_VALUES['year'], step=1)
    age = 2023 - year  # Calculate age based on current year
    
with col2:
    make_options = sorted(list(target_encodings['make'].keys()))
    make = st.selectbox('Make', make_options, make_options.index(DEFAULT_VALUES['make']) if DEFAULT_VALUES['make'] in make_options else 0)
    
    st.write("Additional Information")
    seller_type = st.radio('Seller Type', ['Individual', 'Dealer'], index=0)
    individual = 1 if seller_type == 'Individual' else 0
    
    vehicle_type = st.radio('Vehicle Type', ['Manual', 'Automatic'], index=0)
    vehicle_type = 1 if vehicle_type == 'Manual' else 0
    
    fuel_options = {"Diesel": 1, "Electric": 2, "LPG": 3, "Petrol": 4}
    fuel_type = st.selectbox('Fuel Type', list(fuel_options.keys()), 
                            index=list(fuel_options.values()).index(DEFAULT_VALUES['fuel_type']) if DEFAULT_VALUES['fuel_type'] in fuel_options.values() else 0)
    fuel_type_code = fuel_options[fuel_type]
    
    seat_options = [2, 4, 5, 6, 7, 8, 9]
    seat = st.selectbox('Number of Seats', seat_options, 
                       seat_options.index(DEFAULT_VALUES['seat']) if DEFAULT_VALUES['seat'] in seat_options else 2)

# Create prediction button
if st.button('Predict Price'):
    try:
        # Create a prediction progress placeholder
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        status_text = st.empty()
        
        # Update progress
        status_text.text("Preparing data...")
        progress_bar.progress(10)
        time.sleep(0.2)  # Small delay for visual effect
        
        # Prepare input data
        input_data = pd.DataFrame({
            'km_driven': [km_driven],
            'mileage': [mileage],
            'engine': [engine],
            'max_power': [max_power],
            'age': [age],
            'make': [make],
            'Individual': [individual],
            'Vehical_type': [vehicle_type],
            'fuel_type': [fuel_type_code],
            'seat': [seat]
        })
        
        # Log input data for debugging
        logger.info(f"Input data: {input_data.to_dict()}")
        
        # Update progress
        status_text.text("Loading preprocessing module...")
        progress_bar.progress(25)
        time.sleep(0.3)
        
        # Load the preprocessing module based on selected model
        module_name = MODEL_CONFIG[selected_model]['module_name']
        preprocessing_module = load_preprocessing_module(module_name)
        
        if preprocessing_module is None:
            progress_placeholder.empty()
            status_text.empty()
            st.error(f"Failed to load preprocessing module for {selected_model}")
            st.stop()
        
        # Update progress
        status_text.text("Loading model...")
        progress_bar.progress(40)
        time.sleep(0.3)
        
        # Load the model
        model_path = MODEL_CONFIG[selected_model]['model_path']
        model = load_model(model_path)
        
        if model is None:
            progress_placeholder.empty()
            status_text.empty()
            st.error(f"Failed to load model {model_path}")
            st.stop()
        
        # Update progress
        status_text.text("Preprocessing data...")
        progress_bar.progress(60)
        time.sleep(0.2)
        
        # Preprocess the data
        te = preprocessing_module.target_encoder.transform(input_data)
        sc = preprocessing_module.min_max_scaler_X.transform(te)
        
        # Update progress
        status_text.text("Making prediction...")
        progress_bar.progress(80)
        time.sleep(0.3)
        
        # Make prediction
        scaled_prediction = model.predict(sc)
        
        # Reshape if needed (for neural networks and some models)
        if len(scaled_prediction.shape) == 1:
            scaled_prediction = scaled_prediction.reshape(-1, 1)
        
        # Inverse transform to get actual price
        prediction = preprocessing_module.min_max_scaler_y.inverse_transform(scaled_prediction)
        
        # Update progress
        status_text.text("Finalizing results...")
        progress_bar.progress(95)
        time.sleep(0.2)
        
        # Extract the prediction value
        if isinstance(prediction, np.ndarray):
            if prediction.shape[0] == 1 and prediction.shape[1] == 1:
                prediction_value = prediction[0][0]
            else:
                prediction_value = prediction[0]
        else:
            prediction_value = prediction
        
        # Complete the progress bar
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_placeholder.empty()
        status_text.empty()
            
        # Display prediction
        st.success(f'Predicted Price: ₹{prediction_value:.2f} lakhs')
        
        # Show the encoded data in a collapsible section
        with st.expander("View preprocessed input data"):
            st.write("Target Encoded Data:")
            st.write(pd.DataFrame(te, columns=te.columns if hasattr(te, 'columns') else None))
            st.write("Scaled Data:")
            st.write(pd.DataFrame(sc))
        
        # Show visualization of the prediction
        st.subheader("Price Category")
        if prediction_value > 15:
            st.markdown("🏆 **Luxury Segment** 🏆")
            st.progress(100)
        elif prediction_value > 10:
            st.markdown("✨ **Premium Segment** ✨")
            st.progress(75)
        elif prediction_value > 5:
            st.markdown("🚗 **Mid-range Vehicle** 🚗")
            st.progress(50)
        else:
            st.markdown("🚘 **Entry-level Vehicle** 🚘")
            st.progress(25)
        
        # Additional analysis
        st.subheader("Value Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Age Impact", f"{age} years", 
                     f"{-0.5 * age:.1f}L" if age > 0 else "0L")
        with col2:
            st.metric("Brand Value", make, 
                     f"+{target_encodings['make'][make]:.1f}L" if target_encodings['make'][make] > 5 else f"{target_encodings['make'][make]:.1f}L")
            
    except Exception as e:
        # Clear progress display on error
        if 'progress_placeholder' in locals():
            progress_placeholder.empty()
        if 'status_text' in locals():
            status_text.empty()
            
        logger.error(f"Error during prediction: {e}", exc_info=True)
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please check that all input fields have valid values and the model files exist.")