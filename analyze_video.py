# initalize
import sys
import argparse
import tensorflow as tf
import cv2
import dlib
import numpy as np

from model import OpenNsfwModel
from image_utils import create_yahoo_image_loader
from wide_resnet import WideResNet

# draw labels on video
def draw_label(image2, point, label, rectangle_height,  font = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 0.8, thickness = 1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    if(y>20): # show text above face
        cv2.rectangle(image2, (x, y - size[1]),  (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image2, label, point, font, font_scale, (255, 255, 255), thickness, lineType = cv2.LINE_AA)   
    else: # rectangle is too high on screen to show text above, so show below
        y = y + rectangle_height + size[1]
        cv2.rectangle(image2, (x, y - size[1]),  (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image2, label, (x, y), font, font_scale, (255, 255, 255), thickness, lineType = cv2.LINE_AA)  

def main(argv):
    # parse input
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input video.\
                        Only mp4 is supported.")
    args = parser.parse_args()

    # initialize NSFW Model    
    model = OpenNsfwModel()

    # initialize variables    
    frameTotal=0
    frameNsfw=0
    Nsfw = False
                
    with tf.Session() as sess:

        # set weights and initialize SFW model
        model.build(weights_path="pretrained_models/open_nsfw-weights.npy")
        fn_load_image = None
        fn_load_image = create_yahoo_image_loader()
        sess.run(tf.global_variables_initializer())

        # initialize face detector model and set variables
        detector = dlib.get_frontal_face_detector()
        img_size = 64
        margin = 0.4
        model2 = WideResNet(img_size,  16,  8)()
        model2.load_weights("pretrained_models/weights.28-3.73.hdf5") 

        # load video (argument)
        videoFile = args.input_file
        cap = cv2.VideoCapture(videoFile)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if (ret != True): # if there is no video frame detected then exit
                break
                
            # write video frame to disk and load as an image
            cv2.imwrite('./images/temp.jpg', frame)
            image = fn_load_image('./images/temp.jpg')
            frameTotal= frameTotal+1

            # determine SFW status
            predictions = sess.run(model.predictions, feed_dict={model.input: image})
            if(predictions[0][1]>=0.50):
                frameNsfw= frameNsfw+1
                Nsfw = True
            else:
                Nsfw = False
                    
            # detect faces
            image2 = frame
            image2_h,  image2_w,  _ = np.shape(image2)
            detected = detector(image2, 0)
            faces = np.empty((len(detected),  img_size,  img_size,  3))
            if len(detected) > 0: # one or more faces were found in the frame
                for i, d in enumerate(detected): 
                    # extract the coordinates of the face
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right()+1, d.bottom()+1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w),  0)
                    yw1 = max(int(y1 - margin * h),  0)
                    xw2 = min(int(x2 + margin * w), image2_w - 1)
                    yw2 = min(int(y2 + margin * h), image2_h - 1)
                    # draw a rectangle around the face
                    cv2.rectangle(image2, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    faces[i, :, :, :] = cv2.resize(image2[yw1:yw2+ 1, xw1:xw2 + 1,  :], (img_size,  img_size))
                    # determine the height of the rectangle in case is near top of frame
                    rectangle_height = y2 - y1
                
                # predict ages and genders of faces
                results = model2.predict(faces)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101,  1)
                predicted_ages = results[1].dot(ages).flatten()
                
                # draw predictions by faces
                for i, d in enumerate(detected):
                    label = "{},{}".format(int(predicted_ages[i]), "M" if predicted_genders[i][0] < 0.5 else"F")
                    draw_label(image2, (d.left(), d.top()), label, rectangle_height)
            
            # display whether frame is SFW or not
            if(Nsfw): 
                display_lbl = "NSFW"
            else:
                display_lbl = "SFW"
            size = cv2.getTextSize(display_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
            cv2.rectangle(image2, (1, 20 - size[1]),  (1 + size[0], 20), (255, 0, 0), cv2.FILLED)
            cv2.putText(image2, display_lbl, (1, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, lineType = cv2.LINE_AA)  
            
# display the frame as processed as quickly as possible
            cv2.imshow('frame2',  image2)
            cv2.waitKey(1)
            
# print summary after program runs
        cap.release()
        if(frameNsfw>0):
            print("Contain NSFW")
        else:
            print("SFW")
    if frameTotal>0: print("NSFW % = "+str((frameNsfw/frameTotal)*100))
    else: print("No video frames were detected!")  
if __name__ == "__main__":
    main(sys.argv)
