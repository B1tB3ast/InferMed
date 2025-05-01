from flask import Flask, render_template, request, redirect, url_for, flash
import os
from webtool import process_image_and_text # Import your model function

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For flash messages
# app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder to store uploaded files
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Match the 'name' attribute in your HTML input field
        image = request.files.get('xray_image')
        notes_text = request.form.get('notes_text')

        if image and allowed_file(image.filename):
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)

            # image.save(os.path.join('static', 'uploads', image_filename))

            # Process image + notes
            phenotypes = process_model(image_path, notes_text)

            # Send results to the results page
            return render_template('results.html', phenotypes=phenotypes, xrays=[{'id': 1, 'filename': image.filename}])

        else:
            flash('Invalid or missing image file.', 'danger')

    return render_template('home.html')


# Route for displaying results after analysis
@app.route('/results')
def results():
    return render_template('results.html')


# Placeholder for your model processing function
def process_model(image_path, notes_text):
    # Process the image and notes_text with your model here
    # For now, just returning a dummy list of phenotypes
    return process_image_and_text(image_path, notes_text)

if __name__ == '__main__':
    app.run(debug=True)
