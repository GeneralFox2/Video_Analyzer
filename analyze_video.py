#!/usr/bin/env python3
#python analyze_video.py {video_filename}

# initalize
import sys
import argparse
import tensorflow as tf
import cv2
import dlib
import numpy as np
import detect_and_align
import os

from model import OpenNsfwModel
from image_utils import create_yahoo_image_loader
from wide_resnet import WideResNet
from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.python.platform import gfile

class IdData:
    """Keeps track of known identities and calculates id matches"""

    def __init__(
        self, id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, distance_treshold
    ):
        print("Loading known identities: ", end="")
        self.distance_treshold = distance_treshold
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []
        self.embeddings = None

        image_paths = []
        os.makedirs(id_folder, exist_ok=True)
        ids = os.listdir(os.path.expanduser(id_folder))
        if not ids:
            return

        for id_name in ids:
            id_dir = os.path.join(id_folder, id_name)
            image_paths = image_paths + [os.path.join(id_dir, img) for img in os.listdir(id_dir)]

        print("Found %d images in id folder" % len(image_paths))
        aligned_images, id_image_paths = self.detect_id_faces(image_paths)
        feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)

        if len(id_image_paths) < 5:
            self.print_distance_table(id_image_paths)

    def add_id(self, embedding, new_id, face_patch):
        if self.embeddings is None:
            self.embeddings = np.atleast_2d(embedding)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        self.id_names.append(new_id)
        id_folder = os.path.join(self.id_folder, new_id)
        os.makedirs(id_folder, exist_ok=True)
        filenames = [s.split(".")[0] for s in os.listdir(id_folder)]
        numbered_filenames = [int(f) for f in filenames if f.isdigit()]
        img_number = max(numbered_filenames) + 1 if numbered_filenames else 0
        cv2.imwrite(os.path.join(id_folder, f+{img_number}+".jpg"), face_patch)

    def detect_id_faces(self, image_paths):
        aligned_images = []
        id_image_paths = []
        for image_path in image_paths:
            image = cv2.imread(os.path.expanduser(image_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_patches, _, _ = detect_and_align.detect_faces(image, self.mtcnn)
            if len(face_patches) > 1:
                print(
                    "Warning: Found multiple faces in id image: %s" % image_path
                    + "\nMake sure to only have one face in the id images. "
                    + "If that's the case then it's a false positive detection and"
                    + " you can solve it by increasing the thresolds of the cascade network"
                )
            aligned_images = aligned_images + face_patches
            id_image_paths += [image_path] * len(face_patches)
            path = os.path.dirname(image_path)
            self.id_names += [os.path.basename(path)] * len(face_patches)

        return np.stack(aligned_images), id_image_paths

    def print_distance_table(self, id_image_paths):
        """Prints distances between id embeddings"""
        distance_matrix = pairwise_distances(self.embeddings, self.embeddings)
        image_names = [path.split("/")[-1] for path in id_image_paths]
        print("Distance matrix:\n{:20}".format(""), end="")
        [print("{:20}".format(name), end="") for name in image_names]
        for path, distance_row in zip(image_names, distance_matrix):
            print("\n{:20}".format(path), end="")
            for distance in distance_row:
                print("{:20}".format("%0.3f" % distance), end="")
        print()

    def find_matching_ids(self, embs):
        if self.id_names:
            matching_ids = []
            matching_distances = []
            distance_matrix = pairwise_distances(embs, self.embeddings)
            for distance_row in distance_matrix:
                min_index = np.argmin(distance_row)
                if distance_row[min_index] < self.distance_treshold:
                    matching_ids.append(self.id_names[min_index])
                    matching_distances.append(distance_row[min_index])
                else:
                    matching_ids.append(None)
                    matching_distances.append(None)
        else:
            matching_ids = [None] * len(embs)
            matching_distances = [np.inf] * len(embs)
        return matching_ids, matching_distances

def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print("Loading model filename: %s" % model_exp)
        with gfile.FastGFile(model_exp, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    else:
        raise ValueError("Specify model file, not directory!")

# draw labels on video for Age and Sex detection engine
def draw_label(image2, point, label, rectangle_height,  font = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 0.4, thickness = 1):
    size = cv2.getTextSize(label, font, font_scale + 0.025, thickness)[0]
    x, y = point
    if(y<21): #rectangle is too high on screen to place text on top
        y = y + rectangle_height + size[1]
    cv2.rectangle(image2, (x, y - size[1]),  (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image2, label, (x, y), font, font_scale, (255, 255, 255), thickness, lineType = cv2.LINE_AA)

def main(argv):
    # parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to the input video.")
    parser.add_argument("id_folder", type=str, nargs="+", help="Folder containing ID folders")
    args = parser.parse_args()

    # initialize NSFW Model    
    model = OpenNsfwModel()
    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            
            # set variable defaults
            videoFile = args.input_file
            cap = cv2.VideoCapture(videoFile)
            frameRate = cap.get(5) # get the frame rate
            totalFrameCount = cap.get(7) # get the total number of frames
            img_size = 64
            margin = 0.4
            frameNsfw=0
            isMinor = False
            minorDetected = False
    
            # set weights and initialize SFW model IsSFW
            with tf.variable_scope('IsSFW'):
                model.build(weights_path="pretrained_models/open_nsfw-weights.npy")
                fn_load_image = None
                fn_load_image = create_yahoo_image_loader()
                sess.run(tf.global_variables_initializer())

            # initialize dlib face detector model and set variables
            detector = dlib.get_frontal_face_detector()
            model2 = WideResNet(img_size,  16,  8)()
            model2.load_weights("pretrained_models/weights.29-3.76_utk.hdf5") 
            
            # initialize face identification model
            mtcnn = detect_and_align.create_mtcnn(sess, None)
            load_model("model/20170512-110547.pb")
            threshold = 1.0
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            # Load anchor IDs for face identification model
            id_data = IdData(args.id_folder[0], mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, threshold)
           
            while(cap.isOpened()):
                ret, frame = cap.read()
                frameId = cap.get(1) # get the current frame number
                if (ret != True): # if there is no video frame detected then exit
                    break
                
                # write video frame to disk and load as an image
                cv2.imwrite('./temp_files/temp.jpg', frame)
                image = fn_load_image('./temp_files/temp.jpg')

                # determine SFW status
                predictions = sess.run(model.predictions, feed_dict={model.input: image})
                if(predictions[0][1]>=0.50):
                    frameNsfw= frameNsfw+1
                    display_lbl = "NSFW"
                    AlertColor = [0, 0, 255]
                else:
                    display_lbl = "SFW"
                    AlertColor = [255, 0, 0]
                    
                # detect faces in dlib face detection model
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
                
                    # predict ages and genders of faces using dlib model
                    results = model2.predict(faces)
                    predicted_genders = results[0]
                    ages = np.arange(0, 101).reshape(101,  1)
                    predicted_ages = results[1].dot(ages).flatten()
                
                    # draw predictions by faces using dlib model
                    for i, d in enumerate(detected):
                        isMinor = False
                        if(int(predicted_ages[i]<18)): # detect if a minor is present in the video
                            isMinor = True
                            minorDetected = True
                        label = "{},{},{}".format(int(predicted_ages[i]), "M" if predicted_genders[i][0] < 0.5 else"F", "-MINOR" if isMinor else "")
                        draw_label(image2, (d.left(), d.top()), label, rectangle_height)
                        
                
                 # Locate faces and landmarks in frame for identification
                face_patches, padded_bounding_boxes, landmarks = detect_and_align.detect_faces(frame, mtcnn)
                if len(face_patches) > 0:
                    face_patches = np.stack(face_patches)
                    feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                    embs = sess.run(embeddings, feed_dict=feed_dict)
                    matching_ids, matching_distances = id_data.find_matching_ids(embs)
                    for bb, landmark, matching_id, dist in zip(padded_bounding_boxes, landmarks, matching_ids, matching_distances):
                        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                        cv2.putText(frame, matching_id, (bb[0]+30, bb[3] + 5), font, 1, (255, 0, 255), 1, cv2.LINE_AA)

                # display whether frame is SFW or not
                percentageComplete = round((frameId) / (totalFrameCount) * 100)
                display_lbl = display_lbl + " " + str(percentageComplete) + "% fps= "  + str(round(frameRate, 2))
                size = cv2.getTextSize(display_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(image2, (1, 15 - size[1]),  (1 + size[0], 20), AlertColor, cv2.FILLED)
                cv2.putText(image2, display_lbl, (1, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, lineType = cv2.LINE_AA)  
            
                # display the frame as processed as quickly as possible
                cv2.imshow('frame2',  image2)
                cv2.waitKey(1)
            
            # end of video
            cap.release()
            cv2.destroyAllWindows()
            if os.path.isfile('temp_files/temp.jpg'):
                os.remove("temp_files/temp.jpg")
        
        # print summary
        if totalFrameCount > 0: 
            if(frameNsfw>0):
                if(minorDetected):
                    print("This video contains minors, and " + str(round((frameNsfw / totalFrameCount * 100), 1)) + "% of the video contains NSFW elements.")
                else:
                    print(str(round((frameNsfw / totalFrameCount * 100), 1)) + "% of the video contains NSFW elements.")
            else:
                print("Video is SFW.")
        else: 
            print("No video frames were detected!  Please check the file type or file name.")  

if __name__ == "__main__":
    main(sys.argv)
