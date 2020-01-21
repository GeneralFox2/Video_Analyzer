#!/usr/bin/env python3
#python clean_video.py
import sys
import argparse
import tensorflow as tf
import cv2
import math

from model import OpenNsfwModel
from image_utils import create_yahoo_image_loader

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
    frame_skip = 0    
    
    # load video (argument)
    videoFile = args.input_file
    st = args.input_file
    st = st[:-4]
    outfilename = st + "_Clean.avi"
        
    with tf.Session() as sess:

        # set weights and initialize SFW model
        model.build(weights_path="pretrained_models/open_nsfw-weights.npy")
        fn_load_image = None
        fn_load_image = create_yahoo_image_loader()
        sess.run(tf.global_variables_initializer())
              
        cap = cv2.VideoCapture(videoFile)
        frameRate = cap.get(5) #frame rate
        ret, frame = cap.read()
        height, width, nchannels = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter( outfilename,fourcc, math.floor(frameRate), (width,height))
        global flag
        global frame_skip
        while(True):
            ret, frame = cap.read()
            if (ret != True): # if there is no video frame detected then exit
                break
                
            else:
                cv2.imwrite('./temp_files/temp.jpg', frame)
                image = fn_load_image('./temp_files/temp.jpg')
                frameTotal= frameTotal+1
                predictions = sess.run(model.predictions, feed_dict={model.input: image})
                if(predictions[0][1]<=0.50):
                    out.write(frame)
                else:
                    frameNsfw= frameNsfw+1

# print summary after program runs
        if(frameNsfw>0):
            print("Video contained NSFW content.")
        else:
            print("Video is SFW.")
        print((frameNsfw/frameTotal)*100)
        cap.release()
        out.release()
        
    print("Done")
if __name__ == "__main__":
    main(sys.argv)
