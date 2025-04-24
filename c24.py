import streamlit as st
import pandas as pd
import pickle
from joblib import dump, load
from c24_lr_func import target_encoder, min_max_scaler_X, min_max_scaler_y
from c24_rf_func import target_encoder, min_max_scaler_X, min_max_scaler_y
from c24_nn_func import target_encoder, min_max_scaler_X, min_max_scaler_y

# Set page title
st.title('Vehicle Price Prediction App')
st.write('Enter the details of the vehicle to predict its price')

# Load model with proper error handling
try:
    selected_model = st.selectbox(
    'Select Model',
    options=['Linear Regression', 'Random Forest', 'Neural Network'],
    index=0  # Default to the first option
    )
    
    # model = load('LR_model.pkl')
    # model = pickle.load(open('LR_model.pkl', 'rb'))
    # model = pickle.load(open('Cars24_NN (1)', 'rb'))
    # model = pickle.load(open('C24_RF.pkl', 'rb'))
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Make sure your model file exists and has the correct path")
    model_loaded = False

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

# Create input fields in a more organized layout
st.subheader("Vehicle Details")
col1, col2 = st.columns(2)

with col1:
    km_driven = st.slider("KM Driven", 1000, 100000, step=1000)
    mileage = st.slider("Mileage", 5, 120, step=2)
    engine = st.slider("Engine Power (CC)", 500, 5000, step=100)
    max_power = st.slider("Max Power (HP)", 60, 500, step=20)
    year = st.slider("Year", 2010, 2023, step=1)
    age = 2023 - year  # Calculate age based on current year
    
with col2:
    make_options = sorted(list(target_encodings['make'].keys()))
    make = st.selectbox('Make', make_options)
    # print(make)
    
    st.write("Additional Information")
    seller_type = st.radio('Seller Type', ['individual', 'Dealer'], index=0)
    individual = 1 if seller_type == 'individual' else 0
    
    vehicle_type = st.radio('Vehicle Type', ['manual', 'Automatic'], index=0)
    vehicle_type = 1 if vehicle_type == 'manual' else 0
    
    fuel_options = {"Diesel": 1, "Electric": 2, "LPG": 3, "Petrol": 4}
    fuel_type = st.selectbox('Fuel Type', list(fuel_options.keys()))
    fuel_type = fuel_options[fuel_type]
    
    seat_options = [2, 5, 7, 9]
    seat = st.selectbox('Number of Seats', seat_options)

# Define function for target encoding
def target_encode(df, encodings):
    df_encoded = df.copy()
    # Apply encoding to the 'make' feature
    df_encoded['make'] = df_encoded['make'].map(encodings['make'])
    return df_encoded

# Create prediction button
if st.button('Predict Price'):
    if not model_loaded:
        st.error("Cannot make predictions - model not loaded.")
    else:
        try:
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
                'fuel_type': [fuel_type],
                'seat': [seat]
            })
            # print(input_data)
            
            if selected_model == 'Linear Regression':
                model = load('LR_model.pkl')

                te = target_encoder.transform(input_data)
                sc = min_max_scaler_X.transform(te)
                
                # Show the encoded data in a collapsible section
                with st.expander("View encoded input data"):
                    st.write(te)
                    st.write(sc)
                
                # Make prediction
                scaled_prediction = model.predict(sc)
                prediction = min_max_scaler_y.inverse_transform(scaled_prediction)
                print("prediction for Linear regression",round(prediction[0][0],2),"Lakhs")
                
                # If prediction is a numpy array (sometimes happens with certain models)
                if hasattr(prediction, 'item'):
                    prediction = prediction.item()
                    
                # Display prediction
                st.success(f'Predicted Price: ₹{prediction:.2f} lakhs')
                
                # Optional: Show some interpretation of the result
                if prediction > 10:
                    st.info("This is considered a premium segment vehicle.")
                elif prediction > 5 and prediction <= 10:
                    st.info("This is a mid-range vehicle.")
                else:
                    st.info("This is an entry-level vehicle.")
            
            elif selected_model == 'Random Forest':
                model = load('RF_model.pkl')
                
                te = target_encoder.transform(input_data)
                sc = min_max_scaler_X.transform(te)
                
                # Show the encoded data in a collapsible section
                with st.expander("View encoded input data"):
                    st.write(te)
                    st.write(sc)
                
                # Make prediction
                scaled_prediction = model.predict(sc)
                scaled_prediction = scaled_prediction.reshape(-1, 1)
                
                prediction = min_max_scaler_y.inverse_transform(scaled_prediction)
                print("prediction for Randon Forest",round(prediction[0][0],2),"Lakhs")
                
                # If prediction is a numpy array (sometimes happens with certain models)
                if hasattr(prediction, 'item'):
                    prediction = prediction.item()
                    
                # Display prediction
                st.success(f'Predicted Price: ₹{prediction:.2f} lakhs')
                
                # Optional: Show some interpretation of the result
                if prediction > 10:
                    st.info("This is considered a premium segment vehicle.")
                elif prediction > 5 and prediction <= 10:
                    st.info("This is a mid-range vehicle.")
                else:
                    st.info("This is an entry-level vehicle.")
                
            elif selected_model == 'Neural Network':
                model = load('NN_model.pkl')
                
                te = target_encoder.transform(input_data)
                sc = min_max_scaler_X.transform(te)
                
                # Show the encoded data in a collapsible section
                with st.expander("View encoded input data"):
                    st.write(te)
                    st.write(sc)
                
                # Make prediction
                scaled_prediction = model.predict(sc)
                scaled_prediction = scaled_prediction.reshape(-1, 1)
                
                prediction = min_max_scaler_y.inverse_transform(scaled_prediction)
                print("prediction for Neural Network",round(prediction[0][0],2),"Lakhs")
                
                # If prediction is a numpy array (sometimes happens with certain models)
                if hasattr(prediction, 'item'):
                    prediction = prediction.item()
                    
                # Display prediction
                st.success(f'Predicted Price: ₹{prediction:.2f} lakhs')
                
                # Optional: Show some interpretation of the result
                if prediction > 10:
                    st.info("This is considered a premium segment vehicle.")
                elif prediction > 5 and prediction <= 10:
                    st.info("This is a mid-range vehicle.")
                else:
                    st.info("This is an entry-level vehicle.")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("Please check that all input fields have valid values.")