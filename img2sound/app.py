from flask import (
    Flask, render_template, url_for, redirect, abort, session, \
    g, flash, request)
from werkzeug.utils import secure_filename
import numpy, cv2, click, os
from scipy.io.wavfile import write as wav_write

app = Flask(__name__)
app.debug = True

UPLOAD_FOLDER = './static/uploads/'
SOUNDS_FOLDER = './static/sounds/'
ALLOWED_EXTENSIONS = frozenset(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def build_filters():
    filters = []
    ksize = 31
    for theta in numpy.arange(0, numpy.pi, numpy.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = numpy.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        numpy.maximum(accum, fimg, accum)
    return accum

def main(path, out):
    img = cv2.imread(path)
    data = numpy.random.uniform(-1, 1, 44100)
    filters = build_filters()
    filtered = process(img, filters).swapaxes(0, 1)
    scaled = numpy.int16(filtered / numpy.max(numpy.abs(filtered)) * 32767)
    rv = numpy.ravel(scaled)
    wav_write(out, 44100, rv)

@app.route('/')
def index(): return redirect(url_for('upload'))

@app.route('/upload/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # save to app/uploads/
            filename = 'img.jpg'
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            # generate
            main(path, os.path.join(SOUNDS_FOLDER, 'sound.wav'))
            return redirect(url_for('sound'))
    else:
        return """<!DOCTYPE html>
<html>
  <head> 
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/> 
    <title>Upload Form</title>
  </head>
  <body>
    <p><h1>File Upload</h1></p>
    <form enctype="multipart/form-data" action="/upload/" method="post">
      <input type="file" name="file">
      <br>
      <br>
      <input type="submit" value="Upload">
    </form>
  </body>
</html>"""

@app.route('/sound/')
def sound():
    width = cv2.imread('./static/uploads/img.jpg').shape[0] + 150
    return """<!DOCTYPE html>
<html>
  <head> 
    <title>Sound</title>
  </head>
  <body>
    <p><audio src="/static/sounds/sound.wav" controls style="width:{}px;"></audio></p>
    <img src="/static/uploads/img.jpg" id="upload">
  </body>
</html>""".format(width)

if __name__ == '__main__':
    app.run()