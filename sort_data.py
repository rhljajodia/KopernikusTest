import imaging_interview as img_int
import os
import shutil


# img_int.draw_color_mask()

def display_images():
    return


def get_file_list():
    return


def parse_file_list():
    return


def read_img_pairs():
    return


def get_score():
    return


######################################################################################
#
# sort_camera_views: This function sorts images into separate folders as per the camera id
#
######################################################################################
def sort_camera_views(folder_name:str):
    for (root, dirs, files) in os.walk(folder_name, topdown=True):
        for filename in files:
            working_path = os.path.abspath(folder_name)
            file_src = working_path + '\\' + filename
            file_dest = working_path + '\\' + filename[0:3]  # assuming file naming convention is in 'c##' format
            if not os.path.exists(file_dest):
                os.makedirs(file_dest)
            shutil.move(file_src, file_dest)


data_folder = "dataset"
sort_camera_views(data_folder)  # Only run once to organize data



# C20 uses two different naming convention even though came camera