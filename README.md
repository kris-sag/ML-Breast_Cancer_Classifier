# ML-Breat_Cancer_Classfier
Classification of breast cancer pathology slide images as benign or malignant using PyTorch

# Scripts
• Resnet.py - train a neural network on a set of images. This same script was used for all models, just by switching the models line (ie: models.vgg16, models.resnet152, etc.)
• crossval.py - modified version of Resnet.py file that incorporates cross validation, need to address weight leakage
• Autocropper.py - Code to crop images taken from microscope. Changes the black exterior to white. 
• divide_images.py - Randomly divide a folder of images into testing, training, validation (used 15:70:15). (unchanged and unused for AY 2021-22)
• input_image.py - Determine TP, TN, FP, FN and total accuracy of trained model. 
• image_contrast_graphs.py - Create a graph quanitfiying the contrast of the black and white images

# Model
• VGG16 trained model / state_dict

# Images 
• Photos for Testing - Microcope images before and after autocopping
• contrast_images - Black and white lined images to determine image contrast
• cropped_test - hand cropped images

# Results 
• Graphs - loss and accuracy graphs of different training iterations (descriped in file names, ie: which architecture, with or wihtout cellphone images
• training_data - accuracy and loss after epochs during training of different iterations. 
• Notebooks - resulting notebooks from running various scripts
