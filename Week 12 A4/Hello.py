import os
from flask import Flask, render_template, request

app = Flask(__name__)

# Configure where to save uploaded files
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    # Default image if nothing is uploaded yet
    image_url = "https://images.pexels.com/photos/733416/pexels-photo-733416.jpeg"

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Save the file to the static/uploads folder
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                # Update image_url to show the newly uploaded file
                image_url = f"/{filepath}"

    return render_template('index.html', img=image_url)

if __name__ == '__main__':
    app.run(debug=True)
