from flask import Flask,request,jsonify
from keras.models import load_model
import cv2
import numpy as np

app=Flask(__name__)
model=load_model('models/model-video2text.model')

category_dict={0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

@app.route('/video2text',methods=['GET','POST'])
def video2text():
	
	print('HERE')
	file=request.files["video"]
	file.save('input-video.mp4')

	source=cv2.VideoCapture('input-video.mp4')

	img_size=224
	results=[]
	imgs=[]

	while(True):

		ret,img=source.read()
		if(ret==False):
			break

		img=cv2.resize(img,(0,0),fx=0.25,fy=0.25)

		resized=cv2.resize(img,(img_size,img_size))
		normalized=resized/255.0
		imgs.append(normalized)
	
	result=model.predict(np.array(imgs))
	accs=np.max(result,axis=1)
	labels=np.argmax(result,axis=1)
	#print(accs)
	idxs=accs>0.98
	#print(idxs)

	new_labels=labels[idxs]
	#print(new_labels)
	results=[category_dict[label] for label in new_labels]

	results=np.array(results)
	_,idx=np.unique(results,return_index=True)
	unique=results[np.sort(idx)]
	print(unique)
	occurence=[(results==label).sum() for label in unique]

	max_indxs=np.array(occurence)>1

	selected_unique=unique[max_indxs]

	response={'prediction':str(selected_unique)}
	return jsonify(response)

app.run(debug=True)
