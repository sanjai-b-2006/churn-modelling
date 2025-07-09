import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import base64

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


## streamlit app

# Inject custom CSS for a modern, full-width look
st.markdown(
    '''<style>
    .main .block-container { max-width: 100vw !important; padding-left: 3vw; padding-right: 3vw; }
    .stButton>button { background-color: #4F8BF9; color: white; font-weight: bold; border-radius: 8px; padding: 0.5em 2em; }
    .stButton>button:hover { background-color: #3066be; }
    .stSidebar { background: #f0f2f6; }
    .stSelectbox, .stSlider, .stNumberInput { border-radius: 8px; }
    .stHeader { font-size: 2.5rem; font-weight: 700; color: #4F8BF9; }
    .stSubheader { font-size: 1.5rem; font-weight: 600; }
    </style>''',
    unsafe_allow_html=True
)

# Make the sidebar black with white text for high contrast and readability
st.markdown(
    '''<style>
    section[data-testid="stSidebar"] {
        background: #18191a !important;
        color: #fff !important;
        border-top-right-radius: 16px;
        border-bottom-right-radius: 16px;
        box-shadow: 2px 0 16px rgba(0,0,0,0.12);
        min-width: 270px;
    }
    section[data-testid="stSidebar"] .sidebar-content {
        color: #fff !important;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] h4 {
        color: #fff !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] .stMarkdown ul, section[data-testid="stSidebar"] .stMarkdown li {
        color: #fff !important;
        font-weight: 500;
    }
    .sidebar-logo {
        display: block;
        margin: 0 auto 1.5em auto;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
    }
    </style>''',
    unsafe_allow_html=True
)

# Optional: Add a logo to the sidebar (replace the URL with your bank's logo if desired)
st.sidebar.markdown(
    '<img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" class="sidebar-logo" alt="Bank Logo">',
    unsafe_allow_html=True
)

st.title('Customer Churn Prediction')

# Add a sidebar for app info and instructions
st.sidebar.title('About')
st.sidebar.info('''\
This app predicts the probability of customer churn (exit) for a bank based on user input features.\n\n- Adjust the sliders and select options to input customer data.\n- Click the button below to get the churn probability and prediction.\n\n**Model:** Artificial Neural Network (Keras)\n''')

st.header('Enter Customer Details')

# User input with improved layout
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 30)
    tenure = st.slider('Tenure (years)', 0, 10, 3)
    num_of_products = st.slider('Number of Products', 1, 4, 1)

with col2:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=650)
    balance = st.number_input('Balance', min_value=0.0, value=0.0, step=100.0)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)
    has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
    is_active_member = st.selectbox('Is Active Member', ['No', 'Yes'])

# Convert categorical to numerical for model
has_cr_card_val = 1 if has_cr_card == 'Yes' else 0
is_active_member_val = 1 if is_active_member == 'Yes' else 0

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card_val],
    'IsActiveMember': [is_active_member_val],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Add a button for prediction
if st.button('Predict Churn'):
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
    st.subheader('Results')
    st.write(f'**Churn Probability:** {prediction_proba:.2%}')
    if prediction_proba > 0.5:
        st.error('The customer is likely to churn (exit).')
    else:
        st.success('The customer is not likely to churn (stay).')
