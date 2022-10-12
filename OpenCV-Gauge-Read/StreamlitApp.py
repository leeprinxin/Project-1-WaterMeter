import streamlit as st
import requests

import urllib.request
import cv2
import numpy as np
import time


#requests.get("http://192.168.11.34/control?var=framesize&val=9")
  
threshold_img = st.sidebar.slider("# threshold_img", min_value=1, max_value=255, value=130, help="")
threshold_ln = st.sidebar.slider("# threshold_ln", min_value=1, max_value=255, value=170, help="")
minLineLength = st.sidebar.slider("# minLineLength", min_value=1, max_value=255, value=80, help="")
maxLineGap = st.sidebar.slider("# maxLineGap", min_value=1, max_value=20, value=8, help="")
diff1LowerBound = st.sidebar.slider("# diff1LowerBound", min_value=0.0, max_value=1.0, value=0.1, help="")
diff1UpperBound = st.sidebar.slider("# diff1UpperBound", min_value=0.0, max_value=1.0, value=0.5, help="")
diff2LowerBound = st.sidebar.slider("# diff2LowerBound", min_value=0.0, max_value=1.0, value=0.1, help="")
diff2UpperBound = st.sidebar.slider("# diff2UpperBound", min_value=0.0, max_value=1.0, value=1.0, help="")

startValue = st.sidebar.number_input("# Start Value", min_value=0, max_value=100, value=0)
endValue = st.sidebar.number_input("# End Value", min_value=0, max_value=100, value=50)
startAngle = st.sidebar.number_input("# Start Angle", min_value=0, max_value=360, value=80)
endAngle = st.sidebar.number_input("# End Angle", min_value=0, max_value=360, value=270)


image1 = st.empty()
image2 = st.empty()

col1, col2 = st.columns([1,1])

with col1:
	image3 = st.empty()
with col2:
	image4 = st.empty()

# Test From Local HTTP Server
url = 'http://127.0.0.1:9900/test.jpg'

# Test From Android IP Webcam
#url = 'http://192.168.11.25:8080/shot.jpg'

def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def take_measure(threshold_img, threshold_ln, minLineLength, maxLineGap, diff1LowerBound, diff1UpperBound, diff2LowerBound, diff2UpperBound):
    imgResp=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img1=cv2.imdecode(imgNp,-1)
    img2=cv2.imdecode(imgNp,-1)
    img3 = np.zeros((img2.shape[0], img2.shape[1], 1), dtype = "uint8")
    img4=cv2.imdecode(imgNp,-1)
    img5 = np.zeros((img2.shape[0], img2.shape[1], 1), dtype = "uint8")

    height, width = img1.shape[:2]
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20)

    maxValue = 255
    th, dst2 = cv2.threshold(gray, threshold_img, maxValue, cv2.THRESH_BINARY_INV);
    img3 = dst2.copy()

    if circles is not None:
        a, b, c = circles.shape
        x,y,r = avg_circles(circles, b)
        
        #img5 = img2[y-r:y+r, x-r:x+r]

        #Draw center and circle
        cv2.circle(img1, (x, y), r, (0, 255, 0), 3, cv2.LINE_AA)  # draw circle
        cv2.circle(img1, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle
        
        separation = 10.0 #in degrees
        interval = int(360 / separation)
        p1 = np.zeros((interval,2))  #set empty arrays
        p2 = np.zeros((interval,2))
        p_text = np.zeros((interval,2))
        for i in range(0,interval):
            for j in range(0,2):
                if (j%2==0):
                    p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
                else:
                    p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
        text_offset_x = 10
        text_offset_y = 5
        for i in range(0, interval):
            for j in range(0, 2):
                if (j % 2 == 0):
                    p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                    p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
                else:
                    p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                    p_text[i][j] = y + text_offset_y + 1.2* r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

        #Lines and labels
        for i in range(0,interval):
            cv2.line(img1, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
            cv2.putText(img1, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(255,0,0),1,cv2.LINE_AA)

        cv2.putText(img1, "Gauge OK!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),2,cv2.LINE_AA)
        
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        maxValue = 255
        
        # Threshold image to take better measurements
        th, dst2 = cv2.threshold(gray2, threshold_img, maxValue, cv2.THRESH_BINARY_INV);

        img3 = dst2.copy()

        in_loop = 0
        lines = cv2.HoughLinesP(image=dst2, rho=3, theta=np.pi / 180, threshold=threshold_ln, minLineLength=minLineLength, maxLineGap=maxLineGap)
        final_line_list = []
        
        for line in lines:
            for x1,y1,x2,y2 in line:

                left = x-r
                right = x+r
                top = y-r
                bottom = y+r
                
                if ((x1 > left) and (y1 > top)) and ((x2 < right) and (y2 < bottom)):
                    cv2.line(img4,(x1,y1),(x2,y2),255,1)
                
        #img4 = img5.copy()

        final_line_list = []

        for i in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                
                left = x-r
                right = x+r
                top = y-r
                bottom = y+r

                if ((x1 > left) and (y1 > top)) and ((x2 < right) and (y2 < bottom)):

                    #final_line_list.append([x1, y1, x2, y2])
                    #in_loop = 1

                    diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
                    diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle

                    if (diff1 > diff2):
                        temp = diff1
                        diff1 = diff2
                        diff2 = temp

                    # Check if line is in range of circle
                    #if (((diff1<diff1UpperBound*r) and (diff1>diff1LowerBound*r) and (diff2<diff2UpperBound*r)) and (diff2>diff2LowerBound*r)):
                    if ((diff1<diff1UpperBound*r) and (diff1>diff1LowerBound*r)):
                        line_length = dist_2_pts(x1, y1, x2, y2)
                        final_line_list.append([x1, y1, x2, y2])
                        in_loop = 1

        if (in_loop == 1):
            print(final_line_list)
            x1 = final_line_list[0][0]
            y1 = final_line_list[0][1]
            x2 = final_line_list[0][2]
            y2 = final_line_list[0][3]
            longerline = dist_2_pts(x1,y1,x2,y2)

            for final_line in final_line_list:
                tx1 = final_line[0]
                ty1 = final_line[1]
                tx2 = final_line[2]
                ty2 = final_line[3]
                tlongerline = dist_2_pts(tx1,ty1,tx2,ty2)
                if tlongerline > longerline:
                    x1=tx1
                    y1=ty1
                    x2=tx2
                    y2=ty2

            cv2.line(img2, (x1, y1), (x2, y2), (0, 255, 255), 2)
            dist_pt_0 = dist_2_pts(x, y, x1, y1)
            dist_pt_1 = dist_2_pts(x, y, x2, y2)
            if (dist_pt_0 > dist_pt_1):
                x_angle = x1 - x
                y_angle = y - y1
            else:
                x_angle = x2 - x
                y_angle = y - y2
                
            # Finding angle using the arc tan of y/x
            res = np.arctan(np.divide(float(y_angle), float(x_angle)))

            #Converting to degrees
            res = np.rad2deg(res)
            if x_angle > 0 and y_angle > 0:  #in quadrant I
                final_angle = 270 - res
            if x_angle < 0 and y_angle > 0:  #in quadrant II
                final_angle = 90 - res
            if x_angle < 0 and y_angle < 0:  #in quadrant III
                final_angle = 90 - res
            if x_angle > 0 and y_angle < 0:  #in quadrant IV
                final_angle = 270 - res

            #cv2.putText(img2, "Indicator OK!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),2,cv2.LINE_AA)
            realValue = (final_angle - startAngle) * (endValue - startValue) / (endAngle-startAngle)

            cv2.putText(img2, f"{final_angle:.2f} {realValue:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),2,cv2.LINE_AA)
            print ("Final Angle: ", final_angle)
        else:
            cv2.putText(img2, "Can't find the indicator!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255),2,cv2.LINE_AA)
            
    else:
        cv2.putText(img1, "Can't see the gauge!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,0,255),2,cv2.LINE_AA)
    return img1, img2, img3, img4

while True:
    #Parameters for real gauge
    #min_angle = 45
    #max_angle = 320
    #min_value = 0
    #max_value = 200
    #units = "PSI"
        
    img1, img2, img3, img4 = take_measure(threshold_img, threshold_ln, minLineLength, maxLineGap, diff1LowerBound, diff1UpperBound, diff2LowerBound, diff2UpperBound)
    
    image1.image(img1[:,:,::-1])
    image2.image(img2[:,:,::-1])
    image3.image(img3)
    image4.image(img4[:,:,::-1])
    #image5.image(img5)
    
    time.sleep(5)