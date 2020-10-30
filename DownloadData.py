import os
import wget
import zipfile


def downloadKITTI():
    url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip'
    url1 = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip'
    folder = './KITTI'
    if os.path.exists(folder):
        print("There already exists folder")
    else:
        os.mkdir(folder)
        wget.download(url, folder+'/')
        wget.download(url1, folder + '/')
    
    file_list = os.listdir(folder)

    for i in range(len(file_list)):
        element = file_list[i]
        if '.zip' in element:
            file_name = element
            with zipfile.ZipFile(folder + '/' + file_name, 'r') as tf:
                tf.extractall(folder + '/')
                tf.close()


if __name__ == "__main__":
    downloadKITTI()