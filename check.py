from imutils import paths
import face_recognition
import pickle
import cv2 as cv
import os


class Train:
    def __init__(self, train_path: str):
        self.train_path = train_path

    def run(self):
        # get paths of each file in folder named Images
        # Images here contains my data(folders of various persons)
        image_paths = list(paths.list_images(self.train_path))
        known_encodings = []
        known_names = []
        # loop over the image paths
        for (i, imagePath) in enumerate(image_paths):
            # extract the person name from the image path
            name = imagePath.split(os.path.sep)[-2]
            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv.imread(imagePath)
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            # Use Face_recognition to locate faces
            boxes = face_recognition.face_locations(rgb, model='hog')
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            # loop over the encodings
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(name)
        # save emcodings along with their names in dictionary data
        data = {"encodings": known_encodings, "names": known_names}
        # use pickle to save data into a file for later use
        f = open("face_enc", "wb")
        f.write(pickle.dumps(data))
        f.close()


class Predict:
    def __init__(self, predict_path: str):
        self.predict_path = predict_path

    def run(self):
        # find path of xml file containing haarcascade file
        casc_pathface = os.path.dirname(
            cv.__file__) + "/data/haarcascade_frontalface_alt2.xml"
        # load the harcaascade in the cascade classifier
        face = cv
        face_cascade = cv.CascadeClassifier(casc_pathface)
        # load the known faces and embeddings saved in last file
        data = pickle.loads(open('face_enc', "rb").read())
        # Find path to the image you want to detect face and pass it here
        image = cv.imread(self.predict_path)
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # convert image to Greyscale for haarcascade
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
                                              flags=cv.CASCADE_SCALE_IMAGE)
        # the facial embeddings for face in input
        encodings = face_recognition.face_encodings(rgb)
        names = []
        # loop over the facial embeddings incase
        # we have multiple embeddings for multiple faces
        for encoding in encodings:
            # Compare encodings with encodings in data["encodings"]
            # Matches contain array with boolean values and True for the embeddings it matches closely
            # and False for rest
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            # set name =inknown if no encoding matches
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                # Find positions at which we get True and store them
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face
                for i in matched_idxs:
                    # Check the names at respective indexes we stored in matched_idxs
                    name = data["names"][i]
                    # increase count for the name we got
                    counts[name] = counts.get(name, 0) + 1
                    # set name which has highest count
                    name = max(counts, key=counts.get)

                # update the list of names
                names.append(name)
                # loop over the recognized faces
                for ((x, y, w, h), name) in zip(faces, names):
                    # rescale the face coordinates
                    # draw the predicted face name on the image
                    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(image, name, (x, y), cv.FONT_HERSHEY_SIMPLEX,
                               0.75, (0, 255, 0), 2)
            cv.imshow("Frame", image)

