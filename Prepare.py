import os
import pandas as pd
import numpy as np
import dlib
import time
from PIL import Image
import cv2

def mix_prepare():
    Expression_version = 'Mix'
    Version = '22'

    DATA_FILE = './{}_data_{}.txt'.format(Expression_version, Version)
    LABEL_FILE = './{}_label_{}.txt'.format(Expression_version, Version)
    SUBJECT_FILE = './{}_subject_{}.txt'.format(Expression_version, Version)
    FLOW_FILE = './{}_flow_{}.txt'.format(Expression_version, Version)

    Video_Motion_path = './results/'

    csv_path = './combined_3class_gt.csv'
    data = pd.read_csv(csv_path, header=None)

# # =========================================================================================
    apex_path = './SAMM/SAMM_label.csv'
    SAMM_path = './SAMM/'
    save_path = './SAMM_landmarks_2pic/'
    flow_path = './SAMM_motion_flow/'
    apex = pd.read_csv(apex_path, header=None)
    samm_data = data[data[0] == 'samm']
    for _, row in samm_data.iterrows():
        sub_path = os.path.join(SAMM_path, str(row[1]).zfill(3))
        save_sub_path = os.path.join(save_path, str(row[1]).zfill(3))
        flow_sub_path = os.path.join(flow_path, str(row[1]).zfill(3))

        test_path = os.path.join(sub_path, str(row[2]))
        save_test_path = os.path.join(save_sub_path, str(row[2]))
        flow_test_path = os.path.join(flow_sub_path, str(row[2]))

        pixel = apex[apex[1] == row[2]]

        apex_num = int(pixel[4]) - int(pixel[3])

        pic_list = os.listdir(test_path)
        pic_list.sort()

        pic_apex_name = pic_list[apex_num]
        pic_apex_path = os.path.join(test_path, pic_apex_name)
        save_apex_pic_path = os.path.join(save_test_path, pic_apex_name)

        pic_onset_name = pic_list[0]
        pic_onset_path = os.path.join(test_path, pic_onset_name)
        save_onset_pic_path = os.path.join(save_test_path, pic_onset_name)

        Video_name = 'samm' + '_' + str(row[1]).zfill(3) + '_' + str(row[2]) + '.avi'
        Video_path = os.path.join(Video_Motion_path, Video_name)

        # Output_Video(test_path, Video_path)

        if not os.path.exists(flow_test_path):
            os.makedirs(flow_test_path)

        Flow_pic_path = os.path.join(flow_test_path, 'motion_flow.png')

        print(test_path)

        # Read_Video(Video_path, apex_num, save_onset_pic_path, save_apex_pic_path, Flow_pic_path)

        # Flow_pic_path = os.path.join(flow_test_path, 'Merge.png')

        # if not os.path.exists(flow_test_path):
        #     os.makedirs(flow_test_path)

        with open (FLOW_FILE, 'a') as m:
            m.write(Flow_pic_path + '\n')

        with open (LABEL_FILE, 'a') as l:
            l.write(str(row[3]) + '\n')
        with open (SUBJECT_FILE, 'a') as s:
            s.write('samm ' + str(row[1]).zfill(3) + '\n')
        with open (DATA_FILE, 'a') as dl:
            dl.write(save_onset_pic_path + '\n')
            dl.write(save_apex_pic_path + '\n')

        # if not os.path.exists(save_test_path):
        #     os.makedirs(save_test_path)

        # crop_pic(pic_onset_path, pic_apex_path, save_onset_pic_path, save_apex_pic_path, Flow_pic_path)

# =================================================================================================

    emotion_dict = {'ne': 'negative', 'po': 'positive', 'sur': 'surprise'}

    SMIC_path = './SMIC_all_raw/HS/'
    SMIC_save_path = './SMIC_landmarks/'
    SMIC_flow_path = './SMIC_motion_flow/'
    au_path = './smic_AU.csv'

    SMIC_data = data[data[0] == 'smic']
    au_data = pd.read_csv(au_path)

    for index, (_, row) in enumerate(SMIC_data.iterrows()):

        subject = int(row[1][1:])
        subject_name = 's' + str(subject)
        SMIC_subject = 'smic ' + str(row[1])
        video_name = row[2].split('_')
        emotion = emotion_dict[video_name[1]]
        test_name = subject_name + '_' + video_name[1] + '_' + video_name[2]
        test_path = os.path.join(SMIC_path, subject_name, 'micro', emotion, test_name)
        save_test_path = os.path.join(SMIC_save_path, subject_name, emotion, test_name)
        flow_test_path = os.path.join(SMIC_flow_path, subject_name, emotion, test_name)

        pic_list = os.listdir(test_path)
        pic_list.sort()

        onset_pic = pic_list[0]
        apex_pic = pic_list[int(len(pic_list) / 2)]
        apex_num = int(len(pic_list) / 2)

        onset_path = os.path.join(test_path, onset_pic)
        apex_path = os.path.join(test_path, apex_pic)

        onset_save_path = os.path.join(save_test_path, onset_pic)
        apex_save_path = os.path.join(save_test_path, apex_pic)

        Video_name = 'smic' + '_' + subject_name + '_' + emotion + '_' + test_name + '.avi'
        Video_path = os.path.join(Video_Motion_path, Video_name)

        # Output_Video(test_path, Video_path)

        if not os.path.exists(flow_test_path):
            os.makedirs(flow_test_path)

        Flow_pic_path = os.path.join(flow_test_path, 'motion_flow.png')

        print(test_path)

        # Read_Video(Video_path, apex_num, onset_save_path, apex_save_path, Flow_pic_path)

        with open (DATA_FILE, 'a') as d:
            d.write(onset_save_path + '\n')
            d.write(apex_save_path + '\n')

        with open (LABEL_FILE, 'a') as l:
            l.write(str(row[3]) + '\n')

        with open (SUBJECT_FILE, 'a') as s:
            s.write(SMIC_subject + '\n')

        # Flow_pic_path = os.path.join(flow_test_path, 'Merge.png')

        # if not os.path.exists(flow_test_path):
        #     os.makedirs(flow_test_path)

        with open (FLOW_FILE, 'a') as m:
            m.write(Flow_pic_path + '\n')

        # if not os.path.exists(save_test_path):
        #     os.makedirs(save_test_path)

        # crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, Flow_pic_path)

# ===================================================================================================

    CASMEII_label_file = './CASME2-coding-20140508.xlsx'
    CASMEII_path = './CASME2_RAW_selected/'
    CASMEII_save_path = './CASMEII_2pic/'
    CASMEII_flow_path = './CASMEII_motion_flow/'

    CASMEII_pic_data = pd.read_excel(CASMEII_label_file, header=0)

    CASMEII_data = data[data[0] == 'casme2']

    for index, row in CASMEII_data.iterrows():
        subject, test, label = row[1], row[2], row[3]

        CASMEII_subject = 'casme2 ' + subject

        CASMEII_temp = CASMEII_pic_data[CASMEII_pic_data['Subject'] == int(subject[3:])]
        CASMEII_temp = CASMEII_temp[CASMEII_temp['Filename'] == test]
        CASMEII_onset = 'img' + str(CASMEII_temp.values[0][3]) + '.jpg'
        CASMEII_apex = 'img' + str(CASMEII_temp.values[0][4]) + '.jpg'
        apex_num = CASMEII_temp.values[0][4] - CASMEII_temp.values[0][3]

        CASMEII_test_path = os.path.join(CASMEII_path, subject, test)
        CASMEII_save_test_path = os.path.join(CASMEII_save_path, subject, test)
        CASMEII_flow_test_path = os.path.join(CASMEII_flow_path, subject, test)

        CASMEII_onset_path = os.path.join(CASMEII_test_path, CASMEII_onset)
        CASMEII_onset_save_path = os.path.join(CASMEII_save_test_path, CASMEII_onset)

        CASMEII_apex_path = os.path.join(CASMEII_test_path, CASMEII_apex)
        CASMEII_apex_save_path = os.path.join(CASMEII_save_test_path, CASMEII_apex)

        # if not os.path.exists(CASMEII_save_test_path):
        #     os.makedirs(CASMEII_save_test_path)

        Video_name = 'casme2' + '_' + subject + '_' + test + '.avi'
        Video_path = os.path.join(Video_Motion_path, Video_name)

        # Output_Video(CASMEII_test_path, Video_path)

        if not os.path.exists(CASMEII_flow_test_path):
            os.makedirs(CASMEII_flow_test_path)

        Flow_pic_path = os.path.join(CASMEII_flow_test_path, 'motion_flow.png')

        print(CASMEII_test_path)

        # Read_Video(Video_path, apex_num, CASMEII_onset_save_path, CASMEII_apex_save_path, Flow_pic_path)

        with open (DATA_FILE, 'a') as d:
            d.write(CASMEII_onset_save_path + '\n')
            d.write(CASMEII_apex_save_path + '\n')

        with open (LABEL_FILE, 'a') as l:
            l.write(str(label) + '\n')

        with open (SUBJECT_FILE, 'a') as s:
            s.write(CASMEII_subject + '\n')

        # Flow_pic_path = os.path.join(CASMEII_flow_test_path, 'flow.png')

        # if not os.path.exists(CASMEII_flow_test_path):
        #     os.makedirs(CASMEII_flow_test_path)

        # crop_pic(CASMEII_onset_path, CASMEII_apex_path, CASMEII_onset_save_path, CASMEII_apex_save_path, Flow_pic_path)

        with open (FLOW_FILE, 'a') as m:
            m.write(Flow_pic_path + '\n')

# =======================================================================================================

    image_path = './CK+/cohn-kanade-images/'
    label_path = './CK+/Emotion/'
    save_path = './CK_2pic/'
    au_path = './CK+/FACS/'
    flow_path = './CK_flow/'

    CK_dict = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 2}

    for subject in os.listdir(label_path):
        subject_label_path = os.path.join(label_path, subject)
        subject_image_path = os.path.join(image_path, subject)
        subject_save_path = os.path.join(save_path, subject)
        subject_flow_path = os.path.join(flow_path, subject)
        for test in os.listdir(subject_label_path):
            test_label_path = os.path.join(subject_label_path, test)
            test_image_path = os.path.join(subject_image_path, test)
            test_save_path = os.path.join(subject_save_path, test)
            test_flow_path = os.path.join(subject_flow_path, test)
            if os.listdir(test_label_path):

                with open (os.path.join(test_label_path, os.listdir(test_label_path)[-1]), 'r') as rl:
                    label = rl.readline()
                    label = int(float(label.strip('\n')))

                if label != 0:
                    with open (LABEL_FILE, 'a') as l:
                        l.write(str(CK_dict[label]) + '\n')

                    image_dir = os.listdir(test_image_path)
                    image_dir = [item for item in image_dir if os.path.splitext(item)[1] == '.png']
                    image_dir = sorted(image_dir)

                    onset_path = os.path.join(test_image_path, image_dir[0])
                    onset_save_path = os.path.join(test_save_path, image_dir[0])

                    apex_num = int(len(image_dir) / 2)

                    apex_path = os.path.join(test_image_path, image_dir[apex_num])
                    apex_save_path = os.path.join(test_save_path, image_dir[apex_num])

                    # if not os.path.exists(test_save_path):
                    #     os.makedirs(test_save_path)

                    # image_landmark(onset_path, onset_save_path)
                    # image_landmark(apex_path, apex_save_path)

                    with open (DATA_FILE, 'a') as d:
                        d.write(onset_save_path + '\n')
                        d.write(apex_save_path + '\n')

                    with open (SUBJECT_FILE, 'a') as s:
                        s.write('Macro\n')

                    Flow_pic_path = os.path.join(test_flow_path, 'flow.png')

                    # if not os.path.exists(test_flow_path):
                    #     os.makedirs(test_flow_path)

                    print(test_label_path)

                    crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, Flow_pic_path)

                    with open (FLOW_FILE, 'a') as m:
                        m.write(Flow_pic_path + '\n')


def crop_pic(onset_path, apex_path, onset_save_path, apex_save_path, flow_save_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    onset_image = cv2.imread(onset_path, 1)
    apex_image = cv2.imread(apex_path, 1)

    onset_det = detector(onset_image, 1)[0]
    apex_det = detector(apex_image, 1)[0]

    onset_faces = dlib.full_object_detections()
    apex_faces = dlib.full_object_detections()

    onset_faces.append(predictor(onset_image, onset_det))
    apex_faces.append(predictor(apex_image, apex_det))

    onset_crops = dlib.get_face_chips(onset_image, onset_faces, size=320)
    apex_crops = dlib.get_face_chips(apex_image, apex_faces, size=320)

    onset_crop = onset_crops[0][:280, 50:270]
    apex_crop = apex_crops[0][:280, 50:270]

    onset_cropped = Image.fromarray(cv2.cvtColor(onset_crop, cv2.COLOR_BGR2RGB))
    apex_cropped = Image.fromarray(cv2.cvtColor(apex_crop, cv2.COLOR_BGR2RGB))

    onset_g = cv2.cvtColor(onset_crops[0], cv2.COLOR_RGB2GRAY)
    apex_g = cv2.cvtColor(apex_crops[0], cv2.COLOR_RGB2GRAY)

    pic_size = onset_crops[0].shape
    hsv = np.zeros(pic_size)
    hsv[:,:,1] = cv2.cvtColor(apex_crops[0], cv2.COLOR_RGB2HSV)[:,:,1]

    flow = cv2.calcOpticalFlowFarneback(onset_g, apex_g, flow=None,
                                        pyr_scale=0.5, levels=1, winsize=15,
                                        iterations=2,
                                        poly_n=5, poly_sigma=1.1, flags=0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[:,:,0] = ang * (180/ np.pi / 2)
    hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb_flow = Image.fromarray(rgb_flow.astype('uint8'))

    rgb_flow.save(flow_save_path)
    onset_cropped.save(onset_save_path)
    apex_cropped.save(apex_save_path)

def Output_Video(file_path, save_path):
    fps = 30
    path = file_path
    fourcc = cv2.VideoWriter_fourcc(*'DIVX') # ('D','I','V','X')
    pic_list = os.listdir(path)
    image_pic = cv2.imread(os.path.join(path, pic_list[0]))
    shape = image_pic.shape
    pic_list = [item.strip('.jpg') for item in pic_list]
    pic_list = [item[3:] for item in pic_list]
    pic_list = [int(item) for item in pic_list]
    pic_list.sort()
    shape_video = (shape[1], shape[0])
    videoWriter = cv2.VideoWriter(save_path, fourcc, fps, shape_video)
    for pic in pic_list:
        pic_name = 'img' + str(pic) + '.jpg'
        img12 = cv2.imread(os.path.join(path, pic_name))
        videoWriter.write(img12)
    videoWriter.release()

def Read_Video(video_path, apex_num, onset_save_path, apex_save_path, flow_save_path):
    vc = cv2.VideoCapture(video_path)
    rval, frame = vc.read()
    frames = np.expand_dims(frame, axis=0)
    while rval:
        rval, frame = vc.read()
        if rval:
            frame = np.expand_dims(frame, axis=0)
            frames = np.append(frames, frame, axis=0)
            cv2.waitKey(1)
        else:
            break
    vc.release()
    # return frames

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    onset_image = frames[0]
    apex_image = frames[apex_num]

    onset_image = cv2.cvtColor(onset_image, cv2.COLOR_RGB2BGR)
    apex_image = cv2.cvtColor(apex_image, cv2.COLOR_RGB2BGR)

    onset_det = detector(onset_image, 1)[0]
    apex_det = detector(apex_image, 1)[0]

    onset_faces = dlib.full_object_detections()
    apex_faces = dlib.full_object_detections()

    onset_faces.append(predictor(onset_image, onset_det))
    apex_faces.append(predictor(apex_image, apex_det))

    onset_crops = dlib.get_face_chips(onset_image, onset_faces, size=320)
    apex_crops = dlib.get_face_chips(apex_image, apex_faces, size=320)

    onset_crop = onset_crops[0][:280, 50:270]
    apex_crop = apex_crops[0][:280, 50:270]

    onset_cropped = Image.fromarray(cv2.cvtColor(onset_crop, cv2.COLOR_BGR2RGB))
    apex_cropped = Image.fromarray(cv2.cvtColor(apex_crop, cv2.COLOR_BGR2RGB))

    onset_g = cv2.cvtColor(onset_crops[0], cv2.COLOR_RGB2GRAY)
    apex_g = cv2.cvtColor(apex_crops[0], cv2.COLOR_RGB2GRAY)

    pic_size = onset_crops[0].shape
    hsv = np.zeros(pic_size)
    hsv[:,:,1] = cv2.cvtColor(apex_crops[0], cv2.COLOR_RGB2HSV)[:,:,1]

    flow = cv2.calcOpticalFlowFarneback(onset_g, apex_g, flow=None,
                                        pyr_scale=0.5, levels=1, winsize=15,
                                        iterations=2,
                                        poly_n=5, poly_sigma=1.1, flags=0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[:,:,0] = ang * (180/ np.pi / 2)
    hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb_flow = Image.fromarray(rgb_flow.astype('uint8'))

    rgb_flow.save(flow_save_path)
    onset_cropped.save(onset_save_path)
    apex_cropped.save(apex_save_path)

if __name__ == '__main__':
    start = time.time()
    mix_prepare()
    print('cost: ', time.time() - start)
