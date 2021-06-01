import cv2 
from pynput.keyboard import Controller , Key

"""
-----------Game Link-------------
http://www.trex-game.skipser.com/
"""

# Capture Camera
capture= cv2.VideoCapture(0)
success , frame = capture.read()

#Keyboard variable
kb = Controller()

#region of interest (Sınrılayıcı Kutular)
bounding_box = cv2.selectROI("Draw a Box",frame,False)

x1 = int(bounding_box[0])
y1 = int(bounding_box[1])
x2 = int(bounding_box[0] + bounding_box[2])
y2 = int(bounding_box[1] + bounding_box[3])

cv2.destroyAllWindows()

imageOrj = frame[y1:y2,x1:x2,:] #okuduğumuz framenin x1,x2,y1,y2 çekiyoruz yani seçtiğimiz kutuyu çekiyoruz
bounding_box_Area = imageOrj.shape[0] * imageOrj.shape[1] # sınırlayıcı kutuların alanı 

while True:
    
    success , frame = capture.read()

    image = frame[y1:y2,x1:x2,:]

    # fark görüntüsü (imgOrj görüntüsü - image)
    difference = cv2.subtract(imageOrj,image)
    cv2.imshow("diff",difference)

    ret,th = cv2.threshold(difference[:,:,0],0,150,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

    cv2.imshow("thresh",th)

    #konturun en büyüğü
    c = max(contours,key=cv2.contourArea)

    #konturun değerini yazdırma 
    value = cv2.contourArea(c)/bounding_box_Area# en büyük konturun alanıyla sınırlayıcı kutuların alanına böl
    print(value)

    #konturun değeri 0.2 ve 0.3 arası ise space tuşuna bas 
    if value > 0.3 and value < 0.5  :
        kb.press(Key.space)

    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),2)
    cv2.imshow("Frame" , frame)


    if cv2.waitKey(20) & 0xFF == ord("q"):break


cv2.destroyAllWindows()

