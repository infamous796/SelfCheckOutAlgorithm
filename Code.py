from PIL import Image, ImageDraw
import cv2
import numpy as np


cropping = False

x_start, y_start, x_end, y_end = 0, 0, 0, 0
import keyboard
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:

    _, img = cap.read()
    cv2.imshow("image",img)
    cv2.waitKey(2)
    print("Focus the product  and press s to continue")
    if keyboard.is_pressed("s"):
        break
    cv2.imwrite(r"temp.jpg",img)

image = cv2.imread('temp.jpg')
oriImage = image.copy()

from cvzone.ClassificationModule import Classifier

lab = ["rectangle","elipse"]
myClassif = Classifier("keras_model_shape_Detection.h5","labels_shape_detection.txt")
im = cv2.imread('temp.jpg')
pred = myClassif.getPrediction(im)
ind = pred[1]
res = lab[ind]


def mouse_crop_rec(event, x, y, flags, param):
  
    global x_start, y_start, x_end, y_end, cropping

   
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

   
    elif event == cv2.EVENT_LBUTTONUP:
      
        x_end, y_end = x, y
        cropping = False 

        refPoint = [(x_start, y_start), (x_end, y_end)]

        if len(refPoint) == 2: #when two points were found 

            filename = 'temp.jpg'
            img = Image.open(filename)
            width, height = img.size
            x = (width - height)//2
            img_cropped = img.crop((x_start, y_start, x_end, y_end)) 

            mask = Image.new('L', img_cropped.size)
            mask_draw = ImageDraw.Draw(mask)
            width, height = img_cropped.size
            mask_draw.rectangle((0, 0, width, height), fill=255)

            img_cropped.putalpha(mask)

            img_cropped.save("NewCroppedImage.png")



def mouse_crop_elipse(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping

  
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

 
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

  
    elif event == cv2.EVENT_LBUTTONUP:
       
        x_end, y_end = x, y
        cropping = False
        refPoint = [(x_start, y_start), (x_end, y_end)]

        if len(refPoint) == 2: #when two points were found 

            filename = 'temp.jpg'
            img = Image.open(filename)
            width, height = img.size
            x = (width - height)//2
            img_cropped = img.crop((x_start, y_start, x_end, y_end)) 

            mask = Image.new('L', img_cropped.size)
            mask_draw = ImageDraw.Draw(mask)
            width, height = img_cropped.size
            mask_draw.ellipse((0, 0, width, height), fill=255)

            img_cropped.putalpha(mask)

            img_cropped.save("NewCroppedImage.png")


cv2.namedWindow("image")

if res == "elipse":
    cv2.setMouseCallback("image", mouse_crop_elipse)

else:
    cv2.setMouseCallback("image", mouse_crop_rec)
    

while True:

    i = image.copy()

    if not cropping:
        cv2.imshow("image", image)

    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", i)
    print("Press c to continue")
    if keyboard.is_pressed("c"):
        break
    else:
        _
    cv2.waitKey(1)

cv2.destroyAllWindows()


png = Image.open('NewCroppedImage.png')
#png.show()
png.load() # required for png.split()

background = Image.new("RGB", png.size, (255, 255, 255))
background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

background.save('foo.jpg', 'JPEG', quality=80)
photo = Image.open("foo.jpg")
wb = Image.open("WhiteBackground.jpg")
wb.paste(photo)
wb.save("Pasted.jpg")



from cvzone.ClassificationModule import Classifier
#cap = cv2.VideoCapture(0)
labels = ["Cello Finegrip Ball Pen","ClassMate 30x21 Notebook","Fancy School Pouch"]
myClassifier = Classifier("keras_model.h5","labels.txt")


img = cv2.imread("Pasted.jpg")



predictions = myClassifier.getPrediction(img)
index = predictions[1]
result = labels[index]
print(result)

