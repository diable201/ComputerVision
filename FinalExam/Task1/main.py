import cv2

data_file = "Dataset"
file = open("./"+data_file+"/input_seg.txt", 'r')

# Enter your name
out_file = open("./results/YourName_SecondName_PLACE.txt", 'w') 

for line in file:
	if ".jpg" in line:
		imgPath = "./"+data_file+"/" + line[:-1]
		img = cv2.imread(imgPath)
		line1 = line
	else:
		cords = line[:-1].split(" ")
		y1 = int(cords[0])
		x1 = int(cords[1])
		y2 = int(cords[2])
		x2 = int(cords[3])

		lp_img = img[y1:y2, x1:x2]
		cv2.imshow("LP img", lp_img)
		cv2.waitKey(0)

		###################################
		########## --Work here-- ########## 
		



		# write your result to out_file
		out_file.write(line1)
		out_file.write(line)
		out_file.write("Here should be your result\n")
		###################################
