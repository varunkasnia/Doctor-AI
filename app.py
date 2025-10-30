from flask import Flask, render_template, request, jsonify
import openai
import os
import pandas as pd
from datetime import datetime
import base64
from dotenv import load_dotenv
import json

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define prescription schema for structured output
PRESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "patient_name": {"type": "string"},
        "patient_age": {"type": "string"},
        "doctor_name": {"type": "string"},
        "date": {"type": "string"},
        "diagnosis": {"type": "string"},
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "medicine_name": {"type": "string"},
                    "dosage": {"type": "string"},
                    "frequency": {"type": "string"},
                    "duration": {"type": "string"}
                }
            }
        },
        "instructions": {"type": "string"}
    },
    "required": ["patient_name", "medications"]
}

# Chat conversation history
conversation_history = []

def encode_image(image_path):
    """Encode image to base64 for OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_prescription_data(image_path):
    """Extract prescription data using OpenAI Vision API with structured output"""
    try:
        base64_image = encode_image(image_path)
        
        response = openai.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical prescription parser. Extract all relevant information from the prescription image and return it in structured JSON format."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all information from this medical prescription including patient name, age, doctor name, date, diagnosis, medications (name, dosage, frequency, duration), and any special instructions."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "prescription_extraction",
                    "schema": PRESCRIPTION_SCHEMA,
                    "strict": True
                }
            },
            max_tokens=1000
        )
        
        # Parse the structured JSON response
        prescription_data = json.loads(response.choices[0].message.content)
        return prescription_data
        
    except Exception as e:
        return {"error": str(e)}

def save_to_csv(prescription_data):
    """Save extracted prescription data to CSV file"""
    try:
        # Flatten medications array for CSV storage
        medications_str = "; ".join([
            f"{med.get('medicine_name', 'N/A')} - {med.get('dosage', 'N/A')} - {med.get('frequency', 'N/A')} - {med.get('duration', 'N/A')}"
            for med in prescription_data.get('medications', [])
        ])
        
        csv_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'patient_name': prescription_data.get('patient_name', 'N/A'),
            'patient_age': prescription_data.get('patient_age', 'N/A'),
            'doctor_name': prescription_data.get('doctor_name', 'N/A'),
            'date': prescription_data.get('date', 'N/A'),
            'diagnosis': prescription_data.get('diagnosis', 'N/A'),
            'medications': medications_str,
            'instructions': prescription_data.get('instructions', 'N/A')
        }
        
        df = pd.DataFrame([csv_data])
        
        # Append to existing CSV or create new one
        if os.path.exists('prescriptions.csv'):
            df.to_csv('prescriptions.csv', mode='a', header=False, index=False)
        else:
            df.to_csv('prescriptions.csv', mode='w', header=True, index=False)
        
        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False

@app.route('/')
def index():
    """Main page for prescription scanning"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_prescription():
    """Handle prescription image upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Extract prescription data
    prescription_data = extract_prescription_data(filepath)
    
    if 'error' in prescription_data:
        return jsonify(prescription_data), 500
    
    # Save to CSV
    if save_to_csv(prescription_data):
        return jsonify({
            'success': True,
            'data': prescription_data,
            'message': 'Prescription scanned and saved successfully!'
        })
    else:
        return jsonify({'error': 'Failed to save data to CSV'}), 500

@app.route('/chat')
def chat():
    """Chatbot interface page"""
    return render_template('chat.html')

@app.route('/chat/send', methods=['POST'])
def chat_send():
    """Handle chatbot messages"""
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Create system message for medical context
        messages = [
            {
                "role": "system",
                "content": "You are a helpful medical assistant chatbot. You can answer general health questions, provide information about medications, and help users understand their prescriptions. Always remind users to consult healthcare professionals for medical advice."
            }
        ] + conversation_history
        
        # Get response from OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history
        conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # Keep only last 10 messages to manage context window
        if len(conversation_history) > 10:
            conversation_history.pop(0)
            conversation_history.pop(0)
        
        return jsonify({
            'success': True,
            'message': assistant_message
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat/clear', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    conversation_history.clear()
    return jsonify({'success': True})

@app.route('/prescriptions')
def view_prescriptions():
    """View all saved prescriptions"""
    if os.path.exists('prescriptions.csv'):
        df = pd.read_csv('prescriptions.csv')
        return jsonify(df.to_dict(orient='records'))
    return jsonify([])

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)
