import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import base64
import time

# Load and preprocess the image
def model_predict(image_path):
    model = tf.keras.models.load_model(r"CNN_plantdiseases_model_high.keras")
    img = cv2.imread(image_path)  # read the file and convert into array
    H,W,C = 224,224,3
    img = cv2.resize(img, (H, W)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = np.array(img)
    img = img.astype("float32")  
    img = img / 255.0  # rescaling
    img = img.reshape(1,H,W,C) #reshaping 
    
    prediction = np.argmax(model.predict(img),axis=-1)[0]

    return prediction


def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Load your local background image (adjust the path accordingly)
background_image_path = "Background.png"  # Replace with your local image path

# Convert the image to base64
background_image_base64 = get_base64_image(background_image_path)

# Add custom CSS to set the background image
st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{background_image_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }}
    </style>
""", unsafe_allow_html=True)
 

#Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])
#app_mode = st.sidebar.selectbox("Select Page",["Home"," ","Disease Recognition"])

# import Image from pillow to open images
from PIL import Image
img = Image.open(r"image.jpg")

# display image using streamlit

st.image(img)
st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)
st.markdown("---")

#Main Page
# Main Page
if(app_mode=="HOME"):
    # Add plant names and diseases
    st.markdown("<h2>Plants and their Diseases:</h2>", unsafe_allow_html=True)

    plant_diseases = {
        "Apple": ["Healthy", "Scab", "Black rot", "Cedar Apple Rust"],
        "Blueberry": ["Healthy"],
        "Cherry": ["Healthy", "Powdery Mildew"],
        "Corn": ["Healthy", "Cercospora Leaf Spot / Gray Leaf Spot", "Common Rust", "Northern Leaf Blight"],
        "Grape": ["Healthy", "Black rot", "Esca (Black Measles)", "Leaf Blight (Isariopsis Leaf Spot)"],
        "Orange": ["Haunglongbing (Citrus greening)"],
        "Peach": ["Healthy", "Bacterial Spot"],
        "Pepper": ["Healthy", "Bacterial Spot"],
        "Potato": ["Healthy", "Late Blight"],
        "Raspberry": ["Healthy"],
        "Soybean": ["Healthy"],
        "Squash": ["Powdery Mildew"],
        "Strawberry": ["Healthy", "Leaf Scorch"],
        "Tomato": ["Healthy", "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot", 
                   "Spider Mites", "Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus"]
    }

    # Display plant names and their diseases
    for plant, diseases in plant_diseases.items():
        st.markdown(f"**{plant}** - {', '.join(diseases)}", unsafe_allow_html=True)


    
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    test_image = st.file_uploader("Choose an Image:")
    
   
    if test_image is not None:
        # Define the save path
        save_path = os.path.join(os.getcwd(), test_image.name)
        print(save_path)
        # Save the file to the working directory
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

    if(st.button("Show Image")):
        st.image(test_image, caption="Uploaded Image", use_container_width=True)

    #Predict button
    if(st.button("Predict")):
        with st.spinner('Model is Usain Bolt 🏃‍➡️🏃‍➡️'):
            time.sleep(3)
        st.success("Prediction complete!")
        st.write("Our Prediction")
        result_index=model_predict(save_path)
        print(result_index)
      

        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        
        
        

        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

        # Additional information about the disease
        st.write("### Disease Information:")
        if "healthy" in class_name[result_index].lower():
            st.success("Good news! Your plant appears to be healthy. Continue with your current care routine.")
        else:
            # Split the class name into plant name and disease name
            plant_name, disease_name = class_name[result_index].split('___')
            disease_name = disease_name.replace('_', ' ')
            st.warning(f"The plant seems to be affected by {plant_name} {disease_name}.")
            st.write("Please consult with a local agricultural expert for specific treatment recommendations.")

            # General advice
            st.write("### General Advice:")
            st.write("- Ensure proper watering and nutrition for your plants.")
            st.write("- Maintain good air circulation around plants to reduce fungal diseases.")
            st.write("- Regularly inspect your plants for early signs of disease.")
            st.write("- Practice crop rotation to prevent the buildup of pathogens in the soil.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Developed by Tuhin Kumar Singha Roy with ❤️ for Sustainable Agriculture🪴</p>", unsafe_allow_html=True)