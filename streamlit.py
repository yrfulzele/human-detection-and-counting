import streamlit as st
import cv2
import imutils
import numpy as np
from PIL import Image
import tempfile


HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect(frame):
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    
    person = 1
    for x,y,w,h in bounding_box_cordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        person += 1
    
    cv2.putText(frame, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.putText(frame, f'Total Persons : {person-1}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    cv2.imshow('output', frame)

    return frame



st.header("Computer Vision Project")
st.title("Real-Time Human Counting 🚶 ")

search_choices = ['Images', 'Video', 'Camera']
search_selected = st.sidebar.selectbox("Your choice please: ", search_choices)

if search_selected == 'Images':
    def load_image(img):
        im = Image.open(img)
        input_img = np.array(im)
        return input_img
        
    # Uploading the File to the Page

    st.header("Upload the Image")
    uploadFile = st.file_uploader(label="", type=['jpg', 'png'])

    # Checking the Format of the page
    if uploadFile is not None:
            # Perform your Manupilations (In my Case applying Filters)
            img = load_image(uploadFile)
            st.image(img)
            
            st.write("Image Uploaded Successfully")
            image = imutils.resize(img, width = min(800, img.shape[1])) 

            result_image = detect(image)
            st.image(result_image)
    else:
            st.write("Make sure you image is in JPG/PNG Format.")

elif search_selected == 'Video':
   
   
    uploadFile = st.file_uploader(label="Upload VIDEO", type=["mp4","mpeg"])
        
    tfile = tempfile.NamedTemporaryFile(delete=False) 
 
    tfile.write(uploadFile.read())
    vf= cv2.VideoCapture(tfile.name) 

    st.write('Detecting people...')
    while vf.isOpened():
      #check is True if reading was successful 
        check, frame =  vf.read()

        if check:
            frame = imutils.resize(frame , width=min(800,frame.shape[1]))
            frame = detect(frame)

            key = cv2.waitKey(1)
            if key== ord('q'):
                break
        else:
            break
    vf.release()
    cv2.destroyAllWindows()
         
elif search_selected == 'Camera':
    flag = 0
    st.header("👇 Click on the button to access the camera 👇")
    scan = st.button('Scan the umber of people') 
    st.write(scan)

    if(scan):
        video = cv2.VideoCapture(0)
        st.write('Detecting people...')

        while True:
            ret, frame = video.read()
            cv2.imshow("Frame", frame)

            frame = detect(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                    break
            elif key == ord('s'):
                cv2.imwrite(filename='saved_img.jpg',img =frame)
                video.release()
                img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                flag = 1
                break
          


        video.release()
        cv2.destroyAllWindows()
        if (flag == 1):
            image = Image.open('saved_img.jpg')
            st.image(image)
        else:
            st.write("Make ")
   

     

       

      
      
            









