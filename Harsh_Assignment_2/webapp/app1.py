from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.datastructures import ImmutableMultiDict
import os
from PIL import Image
import cv2
import math
import numpy as np

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

def process_integral_image(image_path):
    def manual_integral_image(img):
        # Convert image to float for precision
        img = img.astype(np.float64)
        
        # Initialize the integral image with zeros
        integral_img = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=np.float64)
        
        # Compute the integral image
        for y in range(1, integral_img.shape[0]):
            for x in range(1, integral_img.shape[1]):
                integral_img[y, x] = img[y - 1, x - 1] + integral_img[y, x - 1] + integral_img[y - 1, x] - integral_img[y - 1, x - 1]
        return integral_img

    def compute_integral_images(rgb_image):
        # Split the channels
        channels = cv2.split(rgb_image)
        
        # Compute the integral image for each channel manually
        integral_images = [manual_integral_image(channel) for channel in channels]
        
        return integral_images
            
    # Load the image
    rgb_image = cv2.imread(image_path)

    # Compute integral images manually
    integral_images = compute_integral_images(rgb_image)
    # display_integral_images(integral_images)

    def normalize_integral_images(integral_images):
        normalized_images = []
        for img in integral_images:
            norm_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            normalized_images.append(norm_img.astype(np.uint8))
        return normalized_images

    final_image = cv2.merge(normalize_integral_images(integral_images))
    cv2.imwrite(UPLOAD_FOLDER+'integral_' + os.path.basename(image_path), final_image)
    return (UPLOAD_FOLDER+'integral_' + os.path.basename(image_path))


def process_image_stitching(image_path):
    def pan_image_stitch(images):
        # Convert images to Gray
        image1 = cv2.cvtColor(images[0], cv2.IMREAD_GRAYSCALE)
        image2 = cv2.cvtColor(images[1], cv2.IMREAD_GRAYSCALE)

        # Initialize the SIFT detector
        sift = cv2.SIFT_create()
        
        def compute_sift_descriptors(img):
            keypoints, descriptors = sift.detectAndCompute(img, None)
            return keypoints, descriptors

        def select_good_matches(matches):
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            good_matches = np.asarray(good_matches)
            return good_matches
        
        keypoints1, descriptors1 = compute_sift_descriptors(image1)
        keypoints2, descriptors2 = compute_sift_descriptors(image2)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        # Select Good Matches
        matches = select_good_matches(matches)

        # Initialize an empty image for drawing keypoints
        image_with_keypoints1 = cv2.drawKeypoints(image1, keypoints1, None)
        image_with_keypoints2 = cv2.drawKeypoints(image2, keypoints2, None)

        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        #Homography matrix
        homography_matrix, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

        warped_image = cv2.warpPerspective(image1, homography_matrix, (image2.shape[1] + image1.shape[1], image2.shape[0]))
        
        # Combine both images
        stitched_image = warped_image.copy()
        stitched_image[0:image2.shape[0], 0:image2.shape[1]] = image2
    
        return stitched_image

    NUM_OF_IMAGES = 3
    images=[]
    for i in range(NUM_OF_IMAGES):
        images.append(cv2.imread(UPLOAD_FOLDER+"pan_images/img"+str(i+1)+".jpeg"))

    stitched_image = images[0]    
    for i in range(1,len(images)):
        images_to_stitch=[stitched_image, images[i]]
        stitched_image = pan_image_stitch(images_to_stitch)
    cv2.imwrite(UPLOAD_FOLDER+'stitching_' + os.path.basename(image_path), stitched_image)
    return (UPLOAD_FOLDER+'stitching_' + os.path.basename(image_path))


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


@app.route('/integral', methods=['GET', 'POST'])
def integral_upload_file():
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
            processed_image_path = process_integral_image(filepath)
            return redirect(url_for('integral_uploaded_file', filename='integral_' + filename))
    return render_template('integralForm.html')

@app.route('/intergal/uploads/<filename>')
def integral_uploaded_file(filename):
    return render_template('integralUpload.html', filename=filename)  


@app.route('/stitching', methods=['GET', 'POST'])
def stitching_upload_file():
    if request.method == 'POST':
        
        if 'file' not in request.files:
            return redirect(request.url)
        print("PRINTING HERE")
        print(request.files)
        imageData = ImmutableMultiDict(request.files)
        print(request.files.getlist('image'))
        print(imageData)
        print(imageData.getlist('input'))
        for item in request.files.getlist('image'):
            print("HERE")
            data = item.read()
            print('len:', len(data))
        file = request.files['file']
        # file_2 = request.files['file_2']
        # file_3 = request.files['file_3']
        
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            processed_image_path = process_image_stitching(file)
            return redirect(url_for('stitching_uploaded_file', filename='stitching_' + filename))
    return render_template('stitchingForm.html')

@app.route('/stitching/uploads/<filename>')
def stitching_uploaded_file(filename):
    return render_template('stitchingUpload.html', filename=filename) 

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
