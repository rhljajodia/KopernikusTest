import imaging_interview as img_int
import os
import cv2

# Define all global variables

CAMERAS = ['c10', 'c20', 'c21', 'c23']      # camera IDs

BLUR_RADIUS = [5, 5]                        # Gaussian blur radius list

MIN_CONTOUR_AREA = 500                      # min contour area

BLACK_MASK = dict()                         # dictionary for black mask for each camera view
BLACK_MASK['c10'] = (0, 13, 0, 0)
BLACK_MASK['c20'] = (0, 28, 0, 0)
BLACK_MASK['c21'] = (0, 30, 0, 0)
BLACK_MASK['c23'] = (0, 35, 0, 0)

IMG_SIZE = (640, 480)                       # image size to resize all images to

DATA_FOLDER = "dataset-candidates-ml\\dataset"  # relative path to dataset

RESIZED_DIR = "resized"                         # folder name to save resized images


def get_file_list(folder_path: str):
    """
    Helper function to retrieve file list in a directory
    """
    return os.listdir(folder_path)


def resize(file_name, img):
    """
    Helper function to resize an image to the size defined by global variable IMG_SIZE.
    This function also filters out unreadable images and images which are smaller than IMG_SIZE.
    """
    # unreadable image
    if img is None:
        print("    Entry ignored: Image " + file_name + " is unreadable.")
        return -1, img

    height, width = img.shape

    # image too small
    if width < IMG_SIZE[0] or height < IMG_SIZE[1]:
        print("    Entry ignored: Image " + file_name + " is too small (" + str(width) + ' x ' + str(height) + ' px).')
        return -1, img

    # image needing resizing
    if width > IMG_SIZE[0] or height > IMG_SIZE[1]:
        img = cv2.resize(img.copy(), IMG_SIZE, interpolation=cv2.INTER_AREA)
    return 1, img


def resize_images(folder_path, file_list):
    """
    This function resizes images to a pre-define size (IMG_SIZE) and saves them in a
    new directory to save computational time while computing scores
    """
    print("Resizing all images...")
    for file in file_list:
        img = cv2.imread(folder_path + '\\' + file, cv2.IMREAD_GRAYSCALE)
        ret, resized_img = resize(file, img)

        if ret == -1:           # if image not resized successfully, continue
            continue

        # save resized image
        new_path = folder_path + '\\' + RESIZED_DIR
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        cv2.imwrite(new_path + '\\' + file, resized_img)
    print("Resizing all images... done")


def parse_file_list(folder_path, file_list):
    # resize images to IMG_SIZE and save to new directory
    resize_images(folder_path, file_list)

    # read dataset list from new directory
    resized_path = folder_path + '\\' + RESIZED_DIR
    resized_file_list = get_file_list(resized_path)

    # define list to hold duplicate file names
    dupli_list = []

    # loop over camera views
    for cam_id in CAMERAS:

        cam_files = [x for x in resized_file_list if cam_id in x]
        if cam_files:
            print("\n"+cam_id+":")

        # turn file paths to absolute paths
        file_paths = [resized_path + '\\' + x for x in cam_files]

        i = -1  # Define the iterator
        # loop over all images in camera view
        while i < len(file_paths) - 1:
            i += 1
            count = 0

            # image 1
            img1_file = file_paths[i]

            # make sure image 1 is not a duplicate of some other image
            if img1_file in dupli_list:
                print("    image " + img1_file[img1_file.rfind('\\')+1:] + " is itself a duplicate... skipped")
                continue

            # read image 1
            img1 = cv2.imread(img1_file, cv2.IMREAD_GRAYSCALE)

            # preprocess
            img1 = img_int.preprocess_image_change_detection(img1, BLUR_RADIUS, BLACK_MASK[cam_id])

            print("    Finding duplicates for image " + img1_file[img1_file.rfind('\\')+1:], end='')

            # loop ahead over dataset
            for img2_file in file_paths[i+1:]:
                # Make sure image 2 has not already been marked as a duplicate
                if img2_file not in dupli_list:

                    # read image 2
                    img2 = cv2.imread(img2_file, cv2.IMREAD_GRAYSCALE)

                    # preprocess
                    img2 = img_int.preprocess_image_change_detection(img2, BLUR_RADIUS, BLACK_MASK[cam_id])

                    # get score
                    score, _, _ = img_int.compare_frames_change_detection(img1, img2, MIN_CONTOUR_AREA)

                    # mark as a duplicate if score is zero
                    if score == 0:
                        dupli_list.append(img2_file)
                        count += 1

            print("... found: " + str(count) + ".")

    dupli_list = [x[x.rfind('\\') + 1:] for x in dupli_list]  # extract image names from list

    # once all duplicates have been marked for each image, delete them from the original dataset
    print("Deleting all duplicates... ", end='')
    for file in file_list:
        if file in dupli_list:
            os.remove(folder_path + '\\' + file)
    print("done")

    # delete resized images and directory
    print("Deleting resized image directory... ", end='')
    for file in resized_file_list:
        os.remove(resized_path+'\\'+file)
    os.removedirs(resized_path)
    print("done")


def main():
    folder_path = os.path.abspath(DATA_FOLDER)
    file_list = get_file_list(folder_path)
    parse_file_list(folder_path, file_list)


if __name__ == '__main__':
    main()
