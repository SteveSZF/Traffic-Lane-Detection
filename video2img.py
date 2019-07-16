import cv2
import os
root = "/workspace/Videos/video_lane_test"
video = "pandora.avi" 
cap = cv2.VideoCapture(os.path.join(root, video))
c=0             
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    #cv2.imshow("capture", frame)
    cv2.imwrite(root + "/pandora/%(idx)05d.jpg" % {'idx':c},frame)
    c=c+1
    #if cv2.waitKey(100) & 0xFF == ord('q'):
    #    break
cap.release()
#cv2.destroyAllWindows()
