import os
import pandas as pd
import feature_extraction as ft
# Get the path of current working directory
path = os.getcwd()
gaborenergy1 = []
gaborentropy1 = []
w1=[]
h1=[]
area1 = []
rectArea1= []
perimeter1 = []
aspectratio1 = []
rectangularity1 = []
circularity1 = []
equi_diameter1 = []
RedChannel = []
GreenChannel = []
BlueChannel = []
red_mean1 = []
blue_var1 = []
green_var1 = []
green_mean1 = []
blue_mean1 = []
red_var1 = []
contrast1 = []
correlation1 = []
inversedifferencemoments1 = []
entropy1 = []
L=[]
LBP = []
extent1=[]
solidity1=[]
hull_area1= []


rk=0

dict1 = {'Label':L,'gaborenergy': gaborenergy1, 'gaborentropy': gaborentropy1,'width':w1,'Height':h1, 'area': area1, 'Rect_Area':rectArea1,'perimeter': perimeter1,'Extent': extent1,
             'Solidity':solidity1,'Hull_Area':hull_area1,'AspectRatio': aspectratio1, 'Rectangularity': rectangularity1, 'Circularity': circularity1,
             'EquiDimeter': equi_diameter1, 'RedMean': red_mean1, 'GreenMean': green_mean1, 'BlueMean': blue_mean1,
             'RedVar': red_var1,'BlueVar': blue_var1,'GreenVar': green_var1, 'contrast': contrast1, 'correlation': correlation1,
             'inverse difference moments': inversedifferencemoments1, 'entropy': entropy1}

df = pd.DataFrame(dict1)
f=open("f1.csv","a")
# saving the dataframe
df.to_csv("Labled_DATAUpdate1.csv")


# Define the path to the dataset
base_path = r"C:\Users\Manohar\Desktop\Tomato leaf disease using SVM\Tamato_leaf_data"

# Get the list of all folders in the dataset
dir_list = os.listdir(base_path)

for i in dir_list:
    folder_path = os.path.join(base_path, i)
    img_mask = os.path.join(folder_path, "*.jpg")
    print(f"Processing images in: {img_mask}")

    # Assign labels based on folder name
    if "Target_Spot" in i:
        Label = 1
    elif "Tomato_mosaic_virus" in i:
        Label = 2
    elif "YellowLeaf" in i :  # Handle variations
        Label = 0
    elif "Bacterial_spot" in i:
        Label = 3
    elif "Early_blight" in i:
        Label = 4
    elif "healthy" in i:
        Label = 5
    elif "Late_blight" in i:
        Label = 6
    elif "Leaf_Mold" in i:
        Label = 7
    elif "Septoria_leaf_spot" in i:
        Label = 8
    elif "Spider_mites_Two_spotted_spider_mite" in i:
        Label = 9
    else:
        print(f"Unknown folder: {i}")
        continue

    # Call fun1 to process images and extract features
    ft.fun1(img_mask, Label)

print("Feature extraction completed. Check 'Labled_DATAUpdate1.csv' for results.")