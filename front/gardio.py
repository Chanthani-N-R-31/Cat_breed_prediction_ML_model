import gradio as gr
import pandas as pd
import numpy as np
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- Data Configuration ---
# API Key is REMOVED to rely primarily on local data, as requested.
# If you decide to add external data back later, uncomment the API sections below.
LOCAL_DATA_PATH = 'cats_breeds.xlsx' 

# Global dictionary to store local breed details (the cache)
LOCAL_BREED_DATA = {}

# --- 1. LOAD ALL NECESSARY FILES ---
print("--- Loading model and supporting files ---")
try:
    model = load_model('cat_breed_classifier_1.keras')
    with open('common_breeds.json', 'r') as f:
        common_breeds = json.load(f)
    with open('class_labels.json', 'r') as f:
        class_labels = json.load(f)

    # Load local data (Excel file) into a dictionary for fast lookup
    if os.path.exists(LOCAL_DATA_PATH):
        # Using pd.read_excel for the .xlsx file
        df_local = pd.read_excel(LOCAL_DATA_PATH)
        
        if 'Breed' in df_local.columns:
            # Create a standardized key (lowercase, no spaces) for dictionary lookup
            df_local['key'] = df_local['Breed'].astype(str).str.lower().str.replace('[ _-]', '', regex=True)
            LOCAL_BREED_DATA = df_local.set_index('key').to_dict('index')
            print(f"✅ Loaded {len(LOCAL_BREED_DATA)} breeds from local Excel cache.")
        else:
            print(f"⚠️ Local file '{LOCAL_DATA_PATH}' is missing the 'Breed' column.")
            
    else:
        print(f"⚠️ Local data file '{LOCAL_DATA_PATH}' not found. No breed characteristics will be available.")

    # Prepare the DataFrame for robust name lookup (assuming cats_cleaned.csv is present)
    df = pd.read_csv('cats_cleaned.csv')
    df['breed_standard'] = df['breed'].str.lower().str.replace('[ _-]', '', regex=True)
    final_df = df.set_index('breed_standard')
    
except Exception as e:
    print(f"Error during file loading: {e}")

# --- 2. EXTERNAL SERVICE CALL FUNCTION (REMOVED/COMMENTED OUT) ---
# Since you requested to remove the API key and not mention external services, 
# this function is commented out. If you re-enable it, remember to use imports 
# (requests) and define CAT_API_KEY.
def get_external_breed_info(breed_name):
    # This function is now a placeholder and will always return data not found
    return {"Data_Source": "Not Found"}


# --- 3. CREATE THE PREDICTION FUNCTION (Core Logic Simplified) ---
def predict_cat_breed(input_image):
    """Predicts breed and retrieves characteristics ONLY from the local Excel file."""
    # --- ML Prediction Logic ---
    img = input_image.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    raw_breed = class_labels[predicted_index].lower()
    predicted_breed_key = raw_breed.replace(' ', '').replace('_', '')
    
    # Standardized name for Display
    try:
        breed_name_for_display = final_df.loc[predicted_breed_key]['breed'].title()
    except KeyError:
        breed_name_for_display = raw_breed.title()
        
    # --- Data Retrieval Strategy: Local Cache ONLY ---
    breed_info = {}
    standardized_lookup_key = breed_name_for_display.lower().replace(' ', '').replace('_', '').replace('-', '')
    
    # 1. Check Local Cache (XLSX)
    if standardized_lookup_key in LOCAL_BREED_DATA:
        breed_info = LOCAL_BREED_DATA[standardized_lookup_key].copy()
        # Set a flag to indicate successful local retrieval
        breed_info['Data_Source'] = "Local Cache"
    else:
        # Set a flag indicating failure to find local data
        breed_info['Data_Source'] = "Not Found"


    # --- Conditional Output Formatting ---
    
    # Check if we have complete information from the local sheet
    is_data_complete = breed_info.get('Data_Source') == "Local Cache"
    
    markdown_output = f"## 🏆 Predicted Breed: **{breed_name_for_display}**\n"
    markdown_output += f"**Confidence Score:** `{confidence:.2f}%`\n\n---\n"

    if is_data_complete:
        # Show all details
        markdown_output += f"### 💡 Breed Details\n"
        markdown_output += f"* **Average Life Span:** `{breed_info.get('Life Span (Years)', 'N/A')} years`\n"
        markdown_output += f"* **Temperament:** {breed_info.get('Temperament', 'N/A')}\n"
        markdown_output += f"* **Origin:** {breed_info.get('Origin', 'N/A')}\n"
        markdown_output += f"* **Weight Range:** `{breed_info.get('Weight (Imperial)', 'N/A')} lbs`\n"
        
        # We assume 'Description' is in the Excel file
        description = breed_info.get('Description', 'No description available in the sheet.')
        markdown_output += f"\n### 📝 Breed Overview\n{description}\n"
    else:
        # Only show the breed name and confidence if lookup failed
        markdown_output += "### 🤷 Trait Information Not Available\n"
        markdown_output += "Detailed characteristics for this breed could not be found in our database."

    return markdown_output

# --- 4. BUILD AND LAUNCH THE GRADIO INTERFACE ---
print("--- Launching Gradio Interface ---")

iface = gr.Interface(
    fn=predict_cat_breed,
    inputs=gr.Image(type="pil", label="Upload a Cat Image"),
    outputs=gr.Markdown(label="🐾 Cat Breed Prediction and Details 📝"), 
    title="🐾 Cat Breed Identifier with Traits",
    description="AI predicts the cat's breed and provides characteristics like lifespan, temperament, and origin from local data.",
    article="<p style='text-align: center'>Machine Learning Project</p>"
)

iface.launch()