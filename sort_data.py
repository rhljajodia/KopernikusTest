import imaging_interview as img_int
import os
import shutil
from PIL import Image
import cv2

PARAMETER_DICT = [
    ['c10',
     [[[640, 480],
       [['Blur radius', [5, 5]],
        ['Black mask', [0, 13, 0, 0]],
        ['Min contour area', [500]]]],
      [[2688, 1520],
       [['Blur radius', [11, 5, 5, 5]],
        ['Black mask', [0, 13, 0, 0]],
        ['Min contour area', [1500]]]]]],
    ['c20',
     [[[1920, 1080],
       [['Blur radius', [11, 5, 5, 5]],
        ['Black mask', [0, 28.5, 0, 0]],
        ['Min contour area', [1250]]]]]],
    ['c21',
     [[[1920, 1080],
       [['Blur radius', [11, 5, 5, 5]],
        ['Black mask', [0, 30.5, 0, 0]],
        ['Min contour area', [2500]]]]]],
    ['c23',
     [[[1920, 1080],
       [['Blur radius', [11, 5, 5, 5]],
        ['Black mask', [0, 35, 0, 0]],
        ['Min contour area', [600]]]]]]]

to_delete = ['c21_2021_03_27__12_53_37.png',     # ignored - image resolution is 10 by 6 px
             'c21_2021_03_27__10_36_36.png',     # ignored - corrupt image
             'c21_2021_04_27__12_44_38.png',     # ignored - outlier resolution: 1200 x 675
             'c21_2021_04_27__12_04_38.png']     # ignored - outlier resolution: 1100 x 619


def sort_into_camera_views(folder_name: str, cam_id_len=3):
    """
    This function sorts images into separate folders as per the camera id.
    Only needs to be run once, unless new images are added.

    folder_name may be a relative or absolute path to local directory where the images are stored.

    camera_id_len defines length of the camera id used in file names
    For eg: file-name = 'c10_036482947.png'
            camera_id_len = 3
            file-name = 'c010_232421321.png'
            camera_id_len = 4
    """
    for (root, dirs, files) in os.walk(folder_name, topdown=True):
        working_path = os.path.abspath(folder_name)
        for filename in files:
            file_src = working_path + '\\' + filename
            file_dest = working_path + '\\' + filename[0:cam_id_len]
            if filename in to_delete:
                os.remove(file_src)
                continue
            if not os.path.exists(file_dest):
                os.makedirs(file_dest)
            if os.path.exists(file_dest):
                shutil.move(file_src, file_dest)


def get_file_list(folder_name: str):
    """

    :param folder_name:
    :return:
    """
    root_path = os.path.abspath(folder_name)
    file_list = []
    dir_paths = []
    for (root, dirs, files) in os.walk(folder_name, topdown=True):
        if not dirs:
            return root + "\\", [files]
        else:
            for dirName in dirs:
                dir_paths.append(root_path + "\\" + dirName + "\\")
                for (_, _, dir_files) in os.walk(dir_paths[-1], topdown=True):
                    file_list.append(sorted(dir_files))
            return dir_paths, file_list


def get_image_dimensions(path):
    width, height = Image.open(path).size
    return width, height


def get_img_parameters(img_dimensions, camera_id):
    width, height = img_dimensions

    cam_dict = [x for x in range(len(PARAMETER_DICT)) if camera_id == PARAMETER_DICT[x][0]][0]
    parameter_idx = [x for x in range(len(PARAMETER_DICT[cam_dict][1]))
                     if [width, height] == PARAMETER_DICT[cam_dict][1][x][0]][0]
    parameter_list = PARAMETER_DICT[cam_dict][1][parameter_idx][1]
    blur_radius_list = \
        [parameter_list[x][1] for x in range(len(parameter_list)) if parameter_list[x][0] == 'Blur radius'][0]
    black_mask = tuple(
        [parameter_list[x][1] for x in range(len(parameter_list)) if parameter_list[x][0] == 'Black mask'][0])
    min_contour_area = \
        [parameter_list[x][1] for x in range(len(parameter_list)) if parameter_list[x][0] == 'Min contour area'][0][0]
    return [blur_radius_list, black_mask, min_contour_area]


def get_score(img1_path, img2_path, parameters):
    img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
    img1 = img_int.preprocess_image_change_detection(img1, parameters[0], parameters[1])
    img2 = img_int.preprocess_image_change_detection(img2, parameters[0], parameters[1])
    return img_int.compare_frames_change_detection(img1, img2, parameters[2])


def parse_image_pairs(dir_path, filenames, parameters):
    ids_to_delete = []
    for i in range(len(filenames) - 1):
        score, _, _ = get_score(dir_path + filenames[i], dir_path + filenames[i + 1], parameters)
        if score == 0:
            ids_to_delete.append(i)
    dupli_files = [filenames[x] for x in ids_to_delete]
    for name in dupli_files:
        os.remove(dir_path + name)
        print("Removed file: " + name)


def parse_file_list(dir_name, files, cam_id_len=3):
    file_dictionary = []
    camera_id = files[0][0:cam_id_len]
    for file in files:
        path = dir_name + file
        width, height = get_image_dimensions(path)
        idx = []
        if file_dictionary:
            idx = [x for x in range(len(file_dictionary)) if [width, height] == file_dictionary[x][0]]
        if not idx:
            file_dictionary.append([[width, height], [file]])
        else:
            file_dictionary[idx[0]][1].append(file)

    for values, filenames in file_dictionary:
        parameters = get_img_parameters(values, camera_id)
        parse_image_pairs(dir_name, filenames, parameters)


DATA_FOLDER = "dataset-candidates-ml\\dataset"
sort_into_camera_views(DATA_FOLDER)  # Only run once to organize data
dir_paths, file_lists = get_file_list(DATA_FOLDER)
for (dir_path, file_list) in zip(dir_paths, file_lists):
    parse_file_list(dir_path, file_list)
print()
