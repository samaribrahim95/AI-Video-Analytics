#Important Libraries
import os
import cv2
import torch
import base64
import numpy as np
import pandas as pd

#Models Libraries
import face_detection 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

#Dash Libraries
import dash
import shutil
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

def social_with_mask_face(filename, model, detector, mask_classifier, threshold_distance):
  #Fetch Video Properties
  cap = cv2.VideoCapture(filename)
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

  if not os.path.exists("Results"):
    os.mkdir("Results")

  # Initialize Output Video Stream
  out_stream = cv2.VideoWriter(
    'Results/Output.mp4',
    cv2.VideoWriter_fourcc(*"avc1"),
    3,
    (int(width),int(height)), isColor=True)

  sec = 0 
  frameRate = 0.33
  while True:
      
    # Capture Frame-by-Frame
    cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    ret, img = cap.read()

    sec = sec + frameRate

    if ret == False: 
      break

    # Get Frame Dimentions
    height, width, channels = img.shape
    results = model([img])

    class_ids = []
    confidences = []
    boxes = []

    outs = results.pandas().xyxy[0][results.pandas().xyxy[0]['confidence'] > 0.5]
    outs = pd.DataFrame(outs[outs['name'] == 'person'])

    for index, out in outs.iterrows():

      # Get Center, Height and Width of the Box
      w = out['xmax'] - out['xmin']
      h = out['ymax'] - out['ymin']

      # Topleft Co-ordinates
      x = out['xmin']
      y = out['ymin']

      boxes.append([int(x), int(y), int(w), int(h)]) 

    # Initialize empty lists for storing Bounding Boxes of People and their Faces
    persons = []
    masked_faces = []
    unmasked_faces = []

    for box in boxes:
      (x, y, w, h) = box

      persons.append([x,y,w,h])

      # Detect Face in the Person
      person_rgb = img[y:y+h,x:x+w,::-1]

      detections = detector.detect(person_rgb)

      # If a Face is Detected
      if detections.shape[0] > 0:

        detection = np.array(detections[0])
        detection = np.where(detection<0,0,detection)

        # Calculating Co-ordinates of the Detected Face
        x1 = x + int(detection[0])
        x2 = x + int(detection[2])
        y1 = y + int(detection[1])
        y2 = y + int(detection[3])

        try :

          # Crop & BGR to RGB
          face_rgb = img[y1:y2,x1:x2,::-1]   

          # Preprocess the Image
          face_arr = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
          face_arr = np.expand_dims(face_arr, axis=0)
          face_arr = preprocess_input(face_arr)

          # Predict if the Face is Masked or Not
          score = mask_classifier.predict(face_arr)

          # Determine and store Results
          if score[0][0]<0.5:
            masked_faces.append([x1,y1,x2,y2])
          else:
            unmasked_faces.append([x1,y1,x2,y2])

        except:
          continue

    # Calculate Coordinates of People Detected and find Clusters using DBSCAN
    person_coordinates = []

    for p in range(len(persons)):
      person_coordinates.append((persons[p][0]+int(persons[p][2]/2),persons[p][1]+int(persons[p][3]/2)))


    if len(persons)>0:
      clustering = DBSCAN(eps=threshold_distance,min_samples=2).fit(person_coordinates)
      isSafe = clustering.labels_
    else:
      isSafe = []

    # Count 
    person_count = len(persons)
    masked_face_count = len(masked_faces)
    unmasked_face_count = len(unmasked_faces)
    safe_count = np.sum((isSafe==-1)*1)
    unsafe_count = person_count - safe_count

    # Show Clusters using Red Lines
    arg_sorted = np.argsort(isSafe)

    for i in range(1,person_count):

      if isSafe[arg_sorted[i]]!=-1 and isSafe[arg_sorted[i]]==isSafe[arg_sorted[i-1]]:
        cv2.line(img,person_coordinates[arg_sorted[i]],person_coordinates[arg_sorted[i-1]],(0,0,255),2)

    # Put Bounding Boxes on People in the Frame
    for p in range(person_count):

      a,b,c,d = persons[p]

      # Green if Safe, Red if UnSafe
      if isSafe[p]==-1:
        cv2.rectangle(img, (a, b), (a + c, b + d), (0,255,0), 3)
      else:
        cv2.rectangle(img, (a, b), (a + c, b + d), (0,0,255), 3)

    # Put Bounding Boxes on Faces in the Frame
    # Green if Safe, Red if UnSafe
    for f in range(masked_face_count):

      a,b,c,d = masked_faces[f]
      cv2.rectangle(img, (a, b), (c,d), (0,255,0), 3)

    for f in range(unmasked_face_count):

      a,b,c,d = unmasked_faces[f]
      cv2.rectangle(img, (a, b), (c,d), (0,0,255), 3)

    # Show Monitoring Status in a Black Box at the Top
    cv2.rectangle(img,(0,0),(width,50),(0,0,0),-1)
    cv2.rectangle(img,(1,1),(width-1,50),(255,255,255),3)

    xpos = 15

    string = "Total People = "+str(person_count)
    cv2.putText(img,string,(xpos,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    xpos += cv2.getTextSize(string,cv2.FONT_HERSHEY_SIMPLEX,1,2)[0][0]

    string = " ( "+str(safe_count) + " Safe "
    cv2.putText(img,string,(xpos,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    xpos += cv2.getTextSize(string,cv2.FONT_HERSHEY_SIMPLEX,1,2)[0][0]

    string = str(unsafe_count)+ " Unsafe ) "
    cv2.putText(img,string,(xpos,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    xpos += cv2.getTextSize(string,cv2.FONT_HERSHEY_SIMPLEX,1,2)[0][0]
    
    string = "( " +str(masked_face_count)+" Masked "+str(unmasked_face_count)+" Unmasked "+\
            str(person_count-masked_face_count-unmasked_face_count)+" Unknown )"
    cv2.putText(img,string,(xpos,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

    # Write Frame to the Output File
    out_stream.write(img)

  # Release Streams
  out_stream.release()
  cap.release()
  cv2.destroyAllWindows()


UPLOAD_DIRECTORY = "Inputs"

# Initialize a Face Detector --> Confidence Threshold can be Adjusted, Greater values would Detect only Clear Face
detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)

# Load Pretrained Face Mask Classfier (Keras Model)
mask_classifier = load_model("Models/ResNet50_Classifier.h5")

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True)
model.classes = [0]

threshold_distance=400


# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
app = dash.Dash(__name__,
  external_stylesheets= [
    'http://127.0.0.1:8887/css/bootstarp/bootstrap.min.css',
    'http://127.0.0.1:8887/css/style.css',
  ],
  external_scripts=[
    'http://127.0.0.1:8887/js/jquery-3.6.0.min.js',
    'http://127.0.0.1:8887/js/popper.min.js',
    'http://127.0.0.1:8887/js/bootstrap/bootstrap.min.js',
    'http://127.0.0.1:8887/js/my-script.js'
  ])


app.layout = html.Div(
  [
    html.Div(
      className="container",
      children=[
        html.H1("COVID-19 Spread Tracker USING AI"),
        html.Hr(className="header-bottom"),
        html.Div(
          className="row",
          children= [
            html.Div(
              className="col-md-6",
              children= [
                html.Div(
                  className='upload-div',
                  children=[
                    dcc.Upload(
                      id="upload-data",
                      className='upload-content',
                      children=html.Div(
                        [
                          html.Img(
                            className='upload-icon',
                            src='http://127.0.0.1:8887/img/cloud-upload.svg'
                          ),
                          html.P("Drag and drop or click to select a file to upload."),
                          html.Button(
                            "Upload Video",
                            className="upload-btn",
                          )
                        ]
                      ),
                    ),
                  ]
                )
              ]
            ),
            html.Div(
              className="col-md-6 output-div",
              children= [
                dcc.Loading(
                  id="loading-2",
                  children=[
                    html.Div(
                      className='text-center',
                      children=[
                        html.Div(id="output-video")
                      ]
                    )
                  ],
                  type="circle",
                ),
              ]
            )
          ]
        ),    
      ]
    )
  ],
)


def save_file(name, content):
  if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

  """Decode and store a file uploaded with Plotly Dash."""
  data = content.encode("utf8").split(b";base64,")[1]
  with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
      fp.write(base64.decodebytes(data))


@app.callback(
    Output("output-video", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filename, uploaded_file_contents):
  if uploaded_filename is not None and uploaded_file_contents is not None:
    if os.path.isdir("Results"):
      shutil.rmtree("Results")
    if os.path.isdir("Inputs"):
      shutil.rmtree("Inputs")

    save_file(uploaded_filename, uploaded_file_contents) #Save uploaded file
    social_with_mask_face('Inputs/'+uploaded_filename, model, detector, mask_classifier, threshold_distance) #Load Model

    return html.Video(
            controls = True,
            id = 'movie_player',
            src = 'http://127.0.0.1:8887/Results/Output.mp4',
            autoPlay=True)
  else:
    return html.Img(src='http://127.0.0.1:8887/img/person.png')

if __name__ == "__main__":
  app.run_server(debug=True)