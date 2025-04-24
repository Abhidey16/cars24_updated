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

# Define model paths
MODEL_CONFIG = {
    'Linear Regression': {
        'model_path': 'LR_model.pkl',
    },
    'Random Forest': {
        'model_path': 'RF_model.pkl',
    },
    'Neural Network': {
        'model_path': 'NN_model.pkl',
    }
}

# Function to load models/encoders/scalers dynamically
@st.cache_resource
def load_pickle_file(file_path):
    try:
        if os.path.exists(file_path):
            logger.info(f"Loading file from {file_path}")
            return load(file_path)
        else:
            logger.error(f"File not found: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return None

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

# Fuel type mapping (added as per your request)
FUEL_TYPE_MAP = {
    'Diesel': 1,
    'Electric': 2,
    'LPG': 3,
    'Petrol': 4
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

# Load preprocessing files
make_encoder = load_pickle_file('make_encoder.pkl')
scaler_x = load_pickle_file('scaler_x.pkl')
scaler_y = load_pickle_file('scaler_y.pkl')

if make_encoder is None or scaler_x is None or scaler_y is None:
    st.error("Could not load preprocessing files. Please check that make_encoder.pkl, scaler_x.pkl, and scaler_y.pkl exist.")

# Create input fields in a more organized layout
st.subheader("Vehicle Details")
col1, col2 = st.columns(2)

with col1:
    km_driven = st.slider("KM Driven", 1000, 100000, DEFAULT_VALUES['km_driven'], step=1000,
                                  help="Total kilometers the car has been driven. Higher values typically indicate more wear and tear.")
    mileage = st.slider("Mileage", 5, 120, DEFAULT_VALUES['mileage'], step=1,
                                help="Fuel efficiency of the car measured in kilometers per liter. Higher values indicate better fuel economy.")
    engine = st.slider("Engine Power (CC)", 500, 5000, DEFAULT_VALUES['engine'], step=100,                       
        help="Engine capacity in cubic centimeters (cc). Larger engines generally produce more power but may consume more fuel.")
    max_power = st.slider("Max Power (HP)", 60, 500, DEFAULT_VALUES['max_power'], step=10,
                                  help="Maximum power output of the engine measured in horsepower (HP). Higher values indicate better performance.")
    year = st.slider("Year", 2010, 2023, DEFAULT_VALUES['year'], step=1,
                     help="Year of manufacture of the vehicle. Newer vehicles typically have better features and technology and higher price.")
    age = 2023 - year  # Calculate age based on current year
    
with col2:
    # Get make options (assuming make_encoder has classes_ attribute, adjust if needed)
    make_options = sorted(make_encoder.classes_) if hasattr(make_encoder, 'classes_') else ['MARUTI', 'HYUNDAI', 'HONDA', 'TATA', 'MAHINDRA']
    make = st.selectbox('Make', make_options, 
                      make_options.index(DEFAULT_VALUES['make']) if DEFAULT_VALUES['make'] in make_options else 0,
                              help="Brand or manufacturer of the vehicle. Different brands have different market values and depreciation rates.")
    
    st.write("Additional Information")
    seller_type = st.radio('Seller Type', ['Individual', 'Dealer'], index=0,
                                   help="Type of seller. 'Individual' means the car is being sold directly by its owner, while 'Dealer' indicates a professional car dealer.")
    individual = 1 if seller_type == 'Individual' else 0
    
    vehicle_type = st.radio('Vehicle Type', ['Manual', 'Automatic'], index=0,
                                    help="Transmission type. Manual transmissions require the driver to shift gears, while automatic transmissions shift gears automatically.")
    vehicle_type = 1 if vehicle_type == 'Manual' else 0
    
    fuel_type = st.selectbox('Fuel Type', list(FUEL_TYPE_MAP.keys()), 
                            index=list(FUEL_TYPE_MAP.values()).index(DEFAULT_VALUES['fuel_type']) if DEFAULT_VALUES['fuel_type'] in FUEL_TYPE_MAP.values() else 0,
                                    help="Type of fuel used by the vehicle. Diesel, Petrol, LPG (Liquefied Petroleum Gas), or Electric. Each has different costs and environmental impacts.")
    fuel_type_code = FUEL_TYPE_MAP[fuel_type]
    
    seat_options = [2, 5, 7, 9]
    seat = st.selectbox('Number of Seats', seat_options, 
                       seat_options.index(DEFAULT_VALUES['seat']) if DEFAULT_VALUES['seat'] in seat_options else 2,
                               help="Total seating capacity of the vehicle. Cars with more seats are typically larger and may have different pricing characteristics.")

# Create prediction button
if st.button('Predict Price'):
    if make_encoder is None or scaler_x is None or scaler_y is None:
        st.error("Cannot make predictions - preprocessing files not loaded.")
    else:
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
            status_text.text("Encoding categorical features...")
            progress_bar.progress(30)
            time.sleep(0.3)
            
            # Encode the 'make' column
            # Save the original make column for display
            original_make = input_data['make'].copy()
            # Transform only the 'make' column
            input_data = make_encoder.transform(input_data)
            
            # Update progress
            status_text.text("Scaling features...")
            progress_bar.progress(50)
            time.sleep(0.3)
            
            # Scale the input data
            scaled_input = scaler_x.transform(input_data)
            
            # Update progress
            status_text.text("Loading model...")
            progress_bar.progress(70)
            # time.sleep(0.3)
            
            # Load the model
            model_path = MODEL_CONFIG[selected_model]['model_path']
            model = load_pickle_file(model_path)
            
            if model is None:
                progress_placeholder.empty()
                status_text.empty()
                st.error(f"Failed to load model {model_path}")
                st.stop()
            
            # Update progress
            status_text.text("Making prediction...")
            progress_bar.progress(85)
            time.sleep(0.3)
            
            # Make prediction
            scaled_prediction = model.predict(scaled_input)
            
            # Reshape if needed (for neural networks and some models)
            if len(scaled_prediction.shape) == 1:
                scaled_prediction = scaled_prediction.reshape(-1, 1)
            
            # Inverse transform to get actual price
            prediction = scaler_y.inverse_transform(scaled_prediction)
            
            # Update progress
            status_text.text("Finalizing results...")
            progress_bar.progress(95)
            # time.sleep(0.2)
            
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
                st.write("Original Data:")
                display_data = input_data.copy()
                display_data['make'] = original_make  # Restore original make for display
                st.write(display_data)
                
                st.write("Encoded and Scaled Data:")
                st.write(pd.DataFrame(scaled_input))
                        
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
                st.metric("Brand Value", make, "Varies by brand")
                
        except Exception as e:
            # Clear progress display on error
            if 'progress_placeholder' in locals():
                progress_placeholder.empty()
            if 'status_text' in locals():
                status_text.empty()
                
            logger.error(f"Error during prediction: {e}", exc_info=True)
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check that all input fields have valid values and the model files exist.")