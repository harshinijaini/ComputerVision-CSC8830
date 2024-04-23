from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
import cv2
import math
import numpy

app = Flask(__name__)

# Set the path 
UPLOAD_FOLDER = 'static/uploads/' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
def process_image(image_path):
    img = cv2.imread(UPLOAD_FOLDER+os.path.basename(image_path))
    object_dist = 300
    camera_matrix = []
    with open(UPLOAD_FOLDER+ '/camera_matrix.txt', 'r') as f:
            for line in f :
                camera_matrix.append([float(num) for num in line.split(' ')])

    fx, fy, Z = camera_matrix[0][0], camera_matrix[1][1], object_dist

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use HoughCircles to detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=250)

    if circles is not None:
        circles = circles[0, :] 
        x, y, r = circles[0]
        center = (int(x), int(y))
        width = int(2 * r)
        height = int(2 * r)
    else:
            print("No circle detected in the image.")
    bbox = (int(x), int(y), width, height)  # Provide the bounding box coordinates

    print(f"fx: {fx}, fy: {fy}, Z: {Z}")
    print(f"Center (x, y): {int(x)}, {int(y)}; Width (w): {width}; Height (h): {height}")

    # img = Image.open(UPLOAD_FOLDER+ 'calibrated_' + os.path.basename(image_path))
       
    processed_image_path = os.path.join(UPLOAD_FOLDER, 'calibrated_' + os.path.basename(image_path))

    def convert_milli_to_inch(x):
        x = x / 10
        return x / 25.4

    x, y, w, h = bbox
    # Calculate image points
    Image_point1x = x
    Image_point1y = y
    Image_point2x = x + w
    Image_point2y = y + h

    cv2.line(img, (Image_point1x, Image_point1y-h//2), (Image_point1x, Image_point2y-h//2), (0, 0, 255), 8)

    # Convert image points to real-world coordinates
    Real_point1x = Z * (Image_point1x / fx)
    Real_point1y = Z * (Image_point1y / fy)
    Real_point2x = Z * (Image_point2x / fx)
    Real_point2y = Z * (Image_point2x / fy)

    print("Real World Co-ordinates: ")
    print("\t", Real_point1x)
    print("\t", Real_point1y)
    print("\t", Real_point2x)
    print("\t", Real_point2y)

    dist = math.sqrt((Real_point2y - Real_point1y) ** 2 + (Real_point2x - Real_point1x) ** 2)

    distance = round(convert_milli_to_inch(dist*2)*10, 2)

    
    cv2.putText(img, str(distance)+" mm", (Image_point1x - 200, (y + h) // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(UPLOAD_FOLDER+'calibrated_' + os.path.basename(image_path), img)
   
    print("\nDiameter of circular object is: {} mm".format(distance))
    return (UPLOAD_FOLDER+'calibrated_' + os.path.basename(image_path))

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            processed_image_path = process_image(filepath)
            return redirect(url_for('uploaded_file', filename='calibrated_' + filename))
    return render_template('form.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print(filename)
    print("PRINTING FILENAME")
    return render_template('upload.html', filename=filename)

    

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
