import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

def FacialExpressionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def Detect2(file_path=None):
    global label1

    if file_path:
        image = cv2.imread(file_path)
    else:
        cap = cv2.VideoCapture(0)
        ret, image = cap.read()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    pred = "Unable to detect"  
    try:
        for (x, y, w, h) in faces:
            fc = gray_image[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
        print("Predicted Emotion is " + pred)
        label1.configure(foreground="#011638", text=pred)
    except Exception as e:
        print("Error detecting emotion:", e)
        label1.configure(foreground="#011638", text=pred)
        
def Detect(frame):
    global label1

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image, 1.3, 5)
    pred = "Unable to detect"  
    try:
        for (x, y, w, h) in faces:
            fc = gray_image[y:y + h, x:x + w]
            roi = cv2.resize(fc, (48, 48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis, :, :, np.newaxis]))]
        print("Predicted Emotion is " + pred)
        label1.configure(foreground="#011638", text=pred)
    except Exception as e:
        print("Error detecting emotion:", e)
        label1.configure(foreground="#011638", text=pred)

def show_Detect_button(file_path):
    try:
        detect_b = Button(top, text="Detect Emotion", command=lambda: Detect2(file_path), padx=10, pady=5)
        detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
        detect_b.place(relx=0.93, rely=0.5, anchor='center')  

    except Exception as e:
        print("Error showing Detect button:", e)

def upload_image():
    try:
        try:
            live_feed_label.after_cancel(_callback_id)
        except Exception as e:
            print("")
        live_feed_label.pack_forget()
        sign_image.pack()
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((400), (400)))  
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        sign_image.pack(side='left', expand='True')  
        label1.configure(text='')
    
        for widget in top.winfo_children():
            if isinstance(widget, Button) and widget["text"] == "Detect Emotion":
                widget.pack_forget()
        show_Detect_button(file_path)
    except Exception as e:
        print("Error uploading image:", e)

def test_live():
    sign_image.pack_forget()
    toggle = True
    try:
        detect_b.pack_forget()
    except Exception as e:
        print("")
    live_feed_label.pack()
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        frame = cv2.resize(frame, (400, 400))  
        img = Image.fromarray(frame)  
        imgtk = ImageTk.PhotoImage(image=img)  
        live_feed_label.imgtk = imgtk  
        live_feed_label.configure(image=imgtk)  
        Detect(frame)  
    if toggle:
        _callback_id = live_feed_label.after(10, test_live)  
    else:
        live_feed_label.after_cancel(_callback_id)
    

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')
toggle = False
_callback_id = None
label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
live_feed_label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top, height=400, width=400)  
detect_b = None
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a1.json", "model_weights1.h5")

EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


button_frame = Frame(top, background='#CDCDCD')

upload = Button(button_frame, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='left', padx=10)  

test_live_button = Button(button_frame, text="Test Live", command=test_live, padx=10, pady=5)
test_live_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
test_live_button.pack(side='left', padx=10)  

button_frame.pack(side='bottom', pady=20)  

heading = Label(top, text='Emotion Detector', font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack(side='top', pady=20)  
label1.pack(side='bottom', expand='True')
live_feed_label.pack(side='left', expand='True')

cap = cv2.VideoCapture(0)

top.mainloop()
