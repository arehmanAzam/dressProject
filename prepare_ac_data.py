import pandas as pd
import numpy as np
import xml.dom.minidom as parser
import os
import glob
import cv2
class PrepareData:
    """
    Make data out of the xml and pass it into autoencoder
    """
    def __init__(self,folder_xml="/home/slashnext/PycharmProjects/dressProject/data/bb_data/Annotations/",
                 folder_images="/home/slashnext/PycharmProjects/dressProject/data/bb_data/Images/",
                 folder_target="/home/slashnext/PycharmProjects/dressProject/data/bb_data/"):
        self.xmls=folder_xml
        self.folder_images=folder_images
        self.target_folder=folder_target

    def parse_xml(self,xml_name="1.xml"):
        doc = parser.parse(xml_name)
        fileName=doc.getElementsByTagName('filename')
        bb_head={ 'xmin':0 , 'ymin':0 , 'xmax':0 , 'ymax':0 }
        bb_ubody={ 'xmin':0 , 'ymin':0 , 'xmax':0 , 'ymax':0 }
        bb_lbody={ 'xmin':0 , 'ymin':0 , 'xmax':0 , 'ymax':0 }
        dict_elements=['xmin','ymin','xmax','ymax']
        for tag in dict_elements:
            if len(doc.getElementsByTagName("object"))>0:
                bb_head[tag]=int(doc.getElementsByTagName("object")[0].getElementsByTagName(tag)[0].firstChild.nodeValue)
            else:
                bb_head={}
            if len(doc.getElementsByTagName("object"))>1:
                bb_ubody[tag]=int(doc.getElementsByTagName("object")[1].getElementsByTagName(tag)[0].firstChild.nodeValue)
            else:
                bb_ubody={}
            if len(doc.getElementsByTagName("object"))>2:
                bb_lbody[tag]=int(doc.getElementsByTagName("object")[2].getElementsByTagName(tag)[0].firstChild.nodeValue)
            else:
                bb_lbody={}
        return bb_head,bb_ubody,bb_lbody,fileName[0].firstChild.nodeValue


    def prepare_Images(self):


        print ("Data Preparation Started \n")
        if not os.path.isdir(self.target_folder+"head"):
            os.mkdir(self.target_folder+"head")
            os.mkdir(self.target_folder + "head/training")
        if not os.path.isdir(self.target_folder+"ubody"):
            os.mkdir(self.target_folder+"ubody")
            os.mkdir(self.target_folder + "ubody/training")

        if not os.path.isdir(self.target_folder+"lbody"):
            os.mkdir(self.target_folder+"lbody")
            os.mkdir(self.target_folder+"lbody/training")

        xml_files=glob.glob(self.xmls+"*.xml")

        # for xml_file in xml_files:
        #     bb_head,bb_ubody,bb_lbody,fileName=self.parse_xml(xml_name=xml_file)
        #
        #     image=cv2.imread(self.folder_images+fileName)
        #     if image is not None:
        #         if len(bb_head) != 0:
        #             head_img = image[bb_head['ymin']:bb_head['ymax'], bb_head['xmin']:bb_head['xmax']].copy()
        #             head_img = cv2.resize(head_img, dsize=(88, 88))
        #             cv2.imwrite(self.target_folder + "head/training/"+fileName,head_img)
        #         if len(bb_ubody) != 0:
        #             ubody_img = image[bb_ubody['ymin']:bb_ubody['ymax'], bb_ubody['xmin']:bb_ubody['xmax']].copy()
        #             ubody_img=cv2.resize(ubody_img,dsize=(128,128))
        #             cv2.imwrite(self.target_folder + "ubody/training/"+fileName,ubody_img)
        #
        #         if len(bb_lbody) != 0:
        #             lbody_img = image[bb_lbody['ymin']:bb_lbody['ymax'], bb_lbody['xmin']:bb_lbody['xmax']].copy()
        #             lbody_img=cv2.resize(lbody_img,dsize=(128,128))
        #             cv2.imwrite(self.target_folder + "lbody/training/"+fileName,lbody_img)
        #
        #     print("File writed ; " + fileName+ "\n")
        head_training_Path=self.target_folder + "head/"
        ubody_training_Path=self.target_folder + "ubody/"
        lbody_training_Path=self.target_folder + "lbody/"
        print ("Data prepared \n")

        return head_training_Path,ubody_training_Path,lbody_training_Path

if __name__ == '__main__':
    data=PrepareData()
    data.prepare_Images()