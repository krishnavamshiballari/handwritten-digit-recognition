#import libraries
import os
import PIL
import cv2
import glob
import numpy as np
from tkinter import *
from PIL import Image , ImageDraw, ImageGrab

#load model

from keras.models import load_model
model = load_model(r'emnist_.h5') 
print("Model load Successfully, Go for the APP")

word_dict = {0:'0',1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z', 36:'a', 37:'b', 38:'c', 39:'d', 40:'e', 41:'f', 42:'g',43:'h', 44:'i', 45:'j', 46:'k', 47:'l', 48:'m', 49:'n', 50:'o', 51:'p', 52:'q', 53:'r', 54:'s', 55:'t', 56:'u', 57:'v', 58:'w', 59:'x', 60:'y', 61:'z'}

def clear_widget():
	global cv
	#To clear a canvas
	cv.delete("all")

def activate_event (event):
	global lastx, lasty
	#<B1-Motion>
	cv.bind('<B1-Motion>', draw_lines)
	lastx, lasty = event.x, event.y

def draw_lines (event):
	global lastx, lasty 
	x, y = event.x, event.y 
	#do the canvas drawings
	cv.create_line((lastx, lasty, x, y),width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
	lastx, lasty = x, y

def Recognize_Digit():
    global image_number
    predictions = []
    percentage = []
    #image_number=0
    filename = f'image_{image_number}.png'
    widget=cv
    #get the widget coordinates
    
    x=root.winfo_rootx()+widget.winfo_x()
    
    y=root.winfo_rooty()+widget.winfo_y()
    x1=x+widget.winfo_width() 
    y1=y+widget.winfo_height()
    #grab the image, crop it according to my requirement and saved it in png format 
    ImageGrab.grab().crop((x,y,x1, y1)).save(filename)
    
    #read the image in color format
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    #convert the image in grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    #applying Otsu thresholding
    ret, th = cv2.threshold (gray, 0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
    #findContour() function helps in extracting the contours from the image.
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    for cnt in contours:
        
        #Get bounding box and extract ROI 
        x,y,w,h = cv2.boundingRect(cnt)
        #Create rectangle
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 1)
        top = int(0.05 * th.shape[0])
        bottom=top
        left = int(0.05 * th.shape[1])
        right=left
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
        #Extract the image ROT 
        roi = th[y-top:y+h+bottom, x-left:x+w+right]
        # resize roi image to 28x28 pixels
        img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        #reshaping the image to support our model input
        img = img.reshape(1,28,28,1)
        #normalizing the image to support our model input
        img = img/255.0
        #its time to predict the result
        pred = model.predict([img])[0]  
        #numpy.argmax(input array) Returns the indices of the maximum values.
        final_pred = np.argmax(pred)
        data = word_dict[final_pred] +' '+ str(int(max(pred)*100))+'%'
        
        #cv2.putText() method is used to draw a text string on image.
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x,y-5), font, fontScale, color, thickness)
    
    #Showing the predicted results on new window.
    cv2.imshow('image', image)
    cv2.waitKey(0)


#create a main window first (named as root).

root = Tk()
root.resizable(0,0)
root.title("Handwritten Digit Recognition GUI App")

#Initialize fev variables

lastx, lasty = None, None
image_number = 0

#create a canvas for drawing

cv = Canvas(root, width=640, height=480, bg='white') 
cv.grid(row=0, column=0, pady=22, sticky=W , columnspan=2 )

#Tkinter provides a powerful mechanism to let you deal with events yourself.

cv.bind('<Button-1>', activate_event )

#Add Buttons and Labels

btn_save = Button(text="Recognize Digit", command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text = "Clear Widget", command = clear_widget) 
button_clear.grid(row=2, column=1, pady=1, padx=1)

#mainloop() is used when your application is ready to run.
root.mainloop()