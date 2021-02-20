import cv2
import math
import sys
import math
import imutils

# Extract reference contour from the image
def get_ref_contour(img):
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    for contour in contours:
        area = cv2.contourArea(contour)
        img_area = img.shape[0] * img.shape[1]
        if 0.05 < area/float(img_area) < 0.8:
            return contour


# Extract all the contours from the image
def get_all_contours(img):
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    return contours


# Extract Hu moments from an image
def get_hu_moments(img):
    # Threshold image
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Calculate Moments
    moments = cv2.moments(img)

    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)

    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))

    return huMoments


def get_distances(img1, img2):
    thresh1, img1 = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)
    thresh2, img2 = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)

    d1 = cv2.matchShapes(img1, img2, cv2.CONTOURS_MATCH_I1, 0)
    d2 = cv2.matchShapes(img1, img2, cv2.CONTOURS_MATCH_I2, 0)
    d3 = cv2.matchShapes(img1, img2, cv2.CONTOURS_MATCH_I3, 0)

    return d1, d2, d3


def find_shape_in_image(img1, img2):
    ref_contour = get_ref_contour(img1)  # Extract the reference contour
    input_contours = get_all_contours(img2)  # Extract all the contours from the input image
    closest_contour = input_contours[0]
    min_dist = sys.maxsize

    # Finding the closest contour
    for contour in input_contours:
        ret = cv2.matchShapes(ref_contour, contour, 1, 0.0)  # Matching the shapes and taking the closest one
        if ret < min_dist:
            min_dist = ret
            closest_contour = contour

    cv2.drawContours(img2, [closest_contour], -1, (0, 255, 0), 3)
    cv2.imshow('Output', img2)
    cv2.waitKey()


def contour_features(img):

    ret, thresh = cv2.threshold(img, 127, 255, 0)
    im, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)

    # Contour area
    area = cv2.contourArea(cnt)

    #Contour perimeter
    perimeter = cv2.arcLength(cnt, True)

    #Contour approximation
    epsilon = 0.1 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    #Convex hull
    hull = cv2.convexHull(cnt)

    #Checking convexity
    k = cv2.isContourConvex(cnt)

    return area, perimeter, approx, hull, k


def center_contour(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0.0:
            cX = 0
            cY = 0
        else:
            cX = int((M["m10"] // M["m00"]))
            cY = int((M["m01"] // M["m00"]))

        # draw the contour and center of the shape on the image
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(img, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(0)


if __name__=='__main__':

    # 1 Calculate moments in any given image and display them as output

    im = cv2.imread("heart1.jpg", cv2.IMREAD_GRAYSCALE) # Read image as grayscale image
    cv2.imshow('Output', im)
    cv2.waitKey()

    hu_moments = get_hu_moments(im)

    print("First moment: ", hu_moments[0])
    print("Second moment: ", hu_moments[1])
    print("Third moment: ", hu_moments[2])
    print("Fourth moment: ", hu_moments[3])
    print("Fifth moment: ", hu_moments[4])
    print("Sixth moment: ", hu_moments[5])
    print("Seventh moment: ", hu_moments[6])

    # 2 Compare distance for similarity between two pictures

    im1 = cv2.imread("heart.png", cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Output', im1)
    cv2.waitKey()
    im2 = cv2.imread("heart1.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Output', im2)
    cv2.waitKey()
    dist = get_distances(im1, im2)
    print("d1: ", dist[0])
    print("d2: ", dist[1])
    print("d3: ", dist[2])

    # 3 Compute additional features in an image

    image = cv2.imread("heart1.jpg", cv2.IMREAD_GRAYSCALE)
    features = contour_features(image)
    print("Contour area: ", features[0])
    print("Contour perimeter: ", features[1])
    print("Curve approximation: ", features[2])
    print("Convex hull: ", features[3])
    print("Is convex: ", features[4])

    # 4 Find center of contour

    image1 = cv2.imread("shapes_and_colors.jpg")
    center_contour(image1)

    image2 = cv2.imread("leaf.jpg")
    center_contour(image2)

    # 5 Shape matching, find shape in first image in another image with more shapes

    img1 = cv2.imread("pizzas.png")
    img2 = cv2.imread("shapes_and_colors.jpg")
    find_shape_in_image(img1, img2)


