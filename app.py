import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import webtool  # Assuming this is where your ML model is.

app = Flask(__name__)

# Folder to save the uploaded files
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed image extensions (e.g., .jpg, .png)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Check if the form is submitted and contains a file
    if 'image' not in request.files:
        return 'No file part', 400
    file = request.files['image']
    
    # If the user did not select a file, the browser may submit an empty file
    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        # Save the file securely in the static/uploads folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get the text input from the form
        text = request.form['text']

        # Save the text to report.txt
        text_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'report.txt')
        with open(text_file_path, 'w') as f:
            f.write(text)
        
        # Pass image and text file path to your model (assumed to be in model.py)
        phenotypes = webtool.process_image_and_text(file_path, text_file_path)

        # Send the processed results back to the template for display
        return render_template('result.html', image_filename=filename, phenotypes=phenotypes)

    return 'Invalid file type', 400

# If needed to serve the file manually (for any reason):
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
