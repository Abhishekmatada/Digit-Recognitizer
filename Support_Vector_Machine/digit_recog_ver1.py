from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc


digits = datasets.load_digits()
features = digits.data 
labels = digits.target

clf = SVC(C=100, gamma = 0.0001)
clf.fit(features, labels)


img = misc.imread("images/8.jpg")
img = misc.imresize(img, (8,8))
img = img.astype(digits.images.dtype)
img = misc.bytescale(img, high=16, low=0)


x_test = []

for eachRow in img:
	for eachPixel in eachRow:
		x_test.append(sum(eachPixel)/3.0)



print(clf.predict([x_test]))

"""to predict the digit edit the JPG (image->mspaint->save->run in cmd)
which is under images folder"""