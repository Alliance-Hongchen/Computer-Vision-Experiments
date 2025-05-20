import numpy as np
import argparse
import cv2
import imutils

# 设置读取图片的路径
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="picture.jpg", help="Path to the image to be scanned") ## 这里修改为要扫描的图片的位置
args = vars(ap.parse_args())

# 加载原图 并对原图进行Resize
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)  # 根据长宽比自动计算另外一边的尺寸进行resize

# 根据处理找到边缘
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# 显示原始图像和边缘检测图像
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 在边缘图像中找到轮廓，只保留最大的轮廓，并初始化屏幕轮廓
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# 在轮廓上循环
for c in cnts:
	# 近似轮廓
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# 如果我们近似的轮廓有四个点，那么我们可以假设我们已经找到了屏幕
	if len(approx) == 4:
		screenCnt = approx
		break

# 显示这张纸的轮廓
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def order_points(pts):
	# 初始化一个坐标列表，该列表将被排序，使列表中的第一个条目位于左上角，第二个条目位于右上角，第一个条目为右下角，第四个条目为左下角
	rect = np.zeros((4, 2), dtype="float32")

	# 左上角的和最小，而右下角的和最大
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算这些点之间的差异，右上角的差异最小，而左下角的差异最大
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# 返回有序坐标
	return rect

def four_point_transform(image, pts):
	# 获得点的一致顺序，并单独打开它们
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算新图像的宽度，即右下角和左下角x坐标或右上角和左上角x坐标之间的最大距离
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# 计算新图像的高度，将是右上角和右下角y坐标或左上角和左下角y座标之间的最大距离
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 有了新图像的尺寸，构建一组目的点以获得图像自上而下的视图，再次按左上、右上、右下和左下的顺序指定点
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype="float32")

	# 计算透视变换矩阵
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回扭曲的图像
	return warped

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# 显示原始图像和扫描图像
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)
