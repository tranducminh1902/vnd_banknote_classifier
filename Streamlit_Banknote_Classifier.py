import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing import image
import numpy as np
from time import sleep
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

menu = ['Display Image', 'Show Video']

classifier =load_model('VND_Banknotes_Classifier_Model_for_Image.h5')
banknote_labels = {0: '1,000 VND', 1: '10,000 VND', 2: '100,000 VND ', 3: '2,000', 4: '20,000 VND', 5: '200,000 VND', 6: '5,000 VND', 7: '50,000 VND', 8: '500,000 VND'}
YOLOweight = 'banknote_yolov3_training_last.weights'
YOLOcfg = 'yolov3_testing.cfg'

#Preprocess and predict image:
def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224))
    img_array=img_to_array(img)
    img_array=np.expand_dims(img,axis = 0)
    predict= classifier.predict(img_array)
    y_class = predict.argmax(axis=-1)
    results = banknote_labels[int(y_class)]
    print(results)
    return results

# Upload and show result:
def make_prediction():
    st.title("Money Classification")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data includes 9 classes of Money"</h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Money", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = r'C:\Users\trand\streamlit_101\uploaded_images'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        
        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success("Tờ tiền này là: "+result)


choice = st.sidebar.selectbox('Choose source for prediction', menu)

if choice=='Display Image':
    make_prediction()


elif choice == 'Show Video':

    # Load Yolo
    net = cv2.dnn.readNet(YOLOweight, YOLOcfg)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    thres = 0.3

    st.title('Open your webcam')
    st.warning('Webcam show on local computer ONLY')
    show = st.checkbox('Show!')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(1) # device 2/2


    while show:
        _, frame = camera.read()
        
        height, width, channels = frame.shape
        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (224, 224), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > thres:
                    # Object detected
                    print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                
                crop_frame = frame[y:y+h,x:x+w]
                try:
                    crop_frame = cv2.resize(frame,(224,224),interpolation=cv2.INTER_AREA)
                except:
                    print('some error')

                if np.sum([crop_frame])!=0:
                    # crop = crop_frame.astype('float')/255.0
                    crop = img_to_array(crop)
                    crop = np.expand_dims(crop,axis=0)
                    # crop = preprocess_input(crop)

                    prediction = classifier.predict(crop)[0]
                    label=banknote_labels[prediction.argmax()]
                    label_position = (x,y)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow('Banknote Classifier',frame)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()