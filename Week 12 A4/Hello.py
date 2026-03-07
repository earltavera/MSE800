from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    # Updated to your Pexels image link
    image_url = "https://images.pexels.com/photos/733416/pexels-photo-733416.jpeg"
    return render_template('index.html', img=image_url)

if __name__ == '__main__':
    app.run(debug=True)