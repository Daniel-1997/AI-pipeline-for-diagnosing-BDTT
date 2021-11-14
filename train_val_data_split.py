import os, shutil
from xml.dom.minidom import parse

def dataset_make(val_case_name=()):

    p_imgs_dir = '/data/ljm/LRCN/case_CT'  # CT with tumors
    p_xmls_dir = '/data/ljm/LRCN/target_boxes'
    Faster_R_CNN_train_dir = '/data/ljm/LRCN/FRCNN_train'
    Faster_R_CNN_val_dir = '/data/ljm/LRCN/FRCNN_val'

    if not os.path.exists(Faster_R_CNN_train_dir):
        os.mkdir(Faster_R_CNN_train_dir)
    elif len(os.listdir(Faster_R_CNN_train_dir)) != 0:
        for item in os.listdir(Faster_R_CNN_train_dir):
            path = os.path.join(Faster_R_CNN_train_dir, item)
            shutil.rmtree(path)

    if not os.path.exists(Faster_R_CNN_val_dir):
        os.mkdir(Faster_R_CNN_val_dir)
    elif len(os.listdir(Faster_R_CNN_val_dir)) != 0:
        for item in os.listdir(Faster_R_CNN_val_dir):
            path = os.path.join(Faster_R_CNN_val_dir, item)
            shutil.rmtree(path)


    train_img_dir = os.path.join(Faster_R_CNN_train_dir, "JPEGImages")
    train_xml_dir = os.path.join(Faster_R_CNN_train_dir, "Annotations")
    val_img_dir = os.path.join(Faster_R_CNN_val_dir, "JPEGImages")
    val_xml_dir = os.path.join(Faster_R_CNN_val_dir, "Annotations")

    os.mkdir(train_img_dir)
    os.mkdir(train_xml_dir)
    os.mkdir(val_img_dir)
    os.mkdir(val_xml_dir)

    patients = os.listdir(p_imgs_dir)

    if len(val_case_name) == 0:
        return None
    val_set = val_case_name
    train_set = [a for a in patients if a not in val_set]
    for patient in train_set:
        img_dir = os.path.join(p_imgs_dir, patient)
        xml_dir = os.path.join(p_xmls_dir, patient + '_label')
        assert len(os.listdir(img_dir)) == len(os.listdir(xml_dir))
        for xml in os.listdir(xml_dir):
            dom = parse(os.path.join(xml_dir, xml))
            data = dom.documentElement
            objects = data.getElementsByTagName('object')
            if len(objects) > 0:
                old_xml_dir = os.path.join(xml_dir, xml)
                new_xml_dir = os.path.join(train_xml_dir, patient[8:] + xml)
                shutil.copy(old_xml_dir, new_xml_dir)

                old_img_dir = os.path.join(img_dir, xml[:-4] + '.jpg')
                new_img_dir = os.path.join(train_img_dir,
                                           patient[8:] + xml[:-4] + '.jpg')
                shutil.copy(old_img_dir, new_img_dir)

    assert len(os.listdir(train_img_dir)) == len(os.listdir(train_xml_dir))

    for patient in val_set:
        img_dir = os.path.join(p_imgs_dir, patient)
        xml_dir = os.path.join(p_xmls_dir, patient + '_label')
        for xml in os.listdir(xml_dir):
            old_xml_dir = os.path.join(xml_dir, xml)
            new_xml_dir = os.path.join(val_xml_dir, patient[8:] + xml)
            shutil.copy(old_xml_dir, new_xml_dir)

            old_img_dir = os.path.join(img_dir, xml[:-4] + '.jpg')
            new_img_dir = os.path.join(val_img_dir,
                                       patient[8:] + xml[:-4] + '.jpg')
            shutil.copy(old_img_dir, new_img_dir)

    assert len(os.listdir(val_img_dir)) == len(os.listdir(val_xml_dir))
    print('num train images with DBD: ', len(os.listdir(train_img_dir)), '\n',
          'num val images with tumors: ', len(os.listdir(val_img_dir)), '\n')

    num_train_DBD, num_val_DBD = 0, 0
    for xml in os.listdir(train_xml_dir):
        xml_dir = os.path.join(train_xml_dir, xml)
        dom = parse(xml_dir)
        data = dom.documentElement
        objects = data.getElementsByTagName('object')
        num_train_DBD += len(objects)

    for xml in os.listdir(val_xml_dir):
        xml_dir = os.path.join(val_xml_dir, xml)
        dom = parse(xml_dir)
        data = dom.documentElement
        objects = data.getElementsByTagName('object')
        num_val_DBD += len(objects)

    print('num DBD in train set: ', num_train_DBD, '\n',
          'num DBD in val set: ', num_val_DBD, '\n')
