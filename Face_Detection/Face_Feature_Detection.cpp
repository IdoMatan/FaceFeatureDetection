#include "pch.h"
#include<opencv2/objdetect/objdetect.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;

void DetectFetures(Mat frame);

// Define the classifier argument and model path
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier mouth_cascade;
CascadeClassifier nose_cascade;

String face_cascade_name = "C:\\OpenCV3\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "C:\\OpenCV3\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";
String mouth_cascade_name = "C:\\OpenCV3\\opencv\\sources\\data\\haarcascades\\haarcascade_mcs_mouth.xml";
String nose_cascade_name = "C:\\OpenCV3\\opencv\\sources\\data\\haarcascades\\haarcascade_mcs_nose.xml";


int main(int argc, const char** argv)
{

	// Load the model
	if (!face_cascade.load(face_cascade_name))
	{
		std::cout << "Error loading face cascade\n";
		return -1;
	};
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		std::cout << "Error loading eyes cascade\n";
		return -1;
	};
	if (!mouth_cascade.load(mouth_cascade_name))
	{
		std::cout << "Error loading mouth cascade\n";
		return -1;
	};
	if (!nose_cascade.load(nose_cascade_name))
	{
		std::cout << "Error loading nose cascade\n";
		return -1;
	};

	// Load the Pic
	Mat frame = imread("albert-einstein.jpg", IMREAD_COLOR);
	
	// Apply the classifier to the frame
	DetectFetures(frame);

	return 0;
}


void DetectFetures(Mat frame)
{
	// Convert the picture to gray scale
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Detect all the faces in the picture
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces); // , 1.1, 1, 0 | CV_HAAR_MAGIC_VAL, Size(100, 100));

	for (size_t i = 0; i < faces.size(); i++)
	{
		// Mark with circle all the faces in the picture
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(0, 0, 255), 4);
		
		Mat faceROI = frame_gray(faces[i]);

		// In each face, detect the wanted fetures
		std::vector<Rect> eyes;
		std::vector<Rect> mouth;
		std::vector<Rect> nose;
		
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2);
		mouth_cascade.detectMultiScale(faceROI, mouth, 1.7, 10, 0|CV_HAAR_FIND_BIGGEST_OBJECT);
		nose_cascade.detectMultiScale(faceROI, nose, 1.3, 5, 0 | CV_HAAR_FIND_BIGGEST_OBJECT);

		// Draw a circle arund the founded feture
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.025);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
		}
		
		for (size_t j = 0; j < mouth.size(); j++)
		{
			Point mouth_center(faces[i].x + mouth[j].x + mouth[j].width / 2, faces[i].y + mouth[j].y + mouth[j].height / 2);
			int radius = cvRound((mouth[j].width + mouth[j].height)*0.025);
			circle(frame, mouth_center, radius, Scalar(255, 255, 0), 4);
		}

		for (size_t j = 0; j < nose.size(); j++)
		{
			Point nose_center(faces[i].x + nose[j].x + nose[j].width / 2, faces[i].y + nose[j].y + nose[j].height / 2);
			int radius = cvRound((nose[j].width + nose[j].height)*0.025);
			circle(frame, nose_center, radius, Scalar(100, 100, 100), 4);
		}
	}
	// Resize the picture only for nicer view 
	Mat resize_frame;
	resize(frame, resize_frame, Size(), 0.5, 0.5);
	imshow("Face feature detected", resize_frame);
	//waitKey(3000);
	waitKey(10);
	imwrite("Face_detection.jpg", frame);
}