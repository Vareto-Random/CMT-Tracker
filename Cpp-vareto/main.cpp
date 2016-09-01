#include <QCoreApplication>

#include "cmt.h"
#include "gui.h"

using namespace cmt;

static string WIN_NAME = "CMT";

int display(Mat im, CMT & cmt)
{
    //Visualize the output
    //It is ok to draw on im itself, as CMT only uses the grayscale image
    for(size_t i = 0; i < cmt.points_active.size(); i++)
    {
        circle(im, cmt.points_active[i], 2, Scalar(255,0,0));
    }

    Point2f vertices[4];
    cmt.bb_rot.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        line(im, vertices[i], vertices[(i+1)%4], Scalar(255,0,0));
    }

    imshow(WIN_NAME, im);

    return cv::waitKey(1);
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    //Create a CMT object
    CMT cmt;

    //Initialization bounding box
    Rect rect;

    //Create window
    cv::namedWindow(WIN_NAME);

    //Load video
    string input_path = "video.mp4";

    VideoCapture cap;
    cap.open(input_path);

    bool show_preview = true;

    int frame = 0;
    //Show preview until key is pressed
    while (show_preview)
    {
        frame++;

        Mat preview;
        cap >> preview;

        screenLog(preview, "Press a key to start selecting an object.");
        imshow(WIN_NAME, preview);

        char k = cv::waitKey(10);
        if (k != -1) {
            show_preview = false;
        }
    }

    //Get initial image
    Mat im0;
    cap >> im0;

    //get bounding box from user
    rect = getRect(im0, WIN_NAME);

    //Convert im0 to grayscale
    Mat im0_gray;
    if (im0.channels() > 1) {
        cvtColor(im0, im0_gray, CV_BGR2GRAY);
    } else {
        im0_gray = im0;
    }

    //Initialize CMT
    cmt.initialize(im0_gray, rect);

    //Main loop
    while (true) {
        frame++;
        Mat im;
        cap >> im;

         //Exit at end of video stream
         if (im.empty())
             break;

         Mat im_gray;
         if (im.channels() > 1) {
             cvtColor(im, im_gray, CV_BGR2GRAY);
         } else {
             im_gray = im;
         }

         //Let CMT process the frame
         cmt.processFrame(im_gray);
         display(im, cmt);
    }

    return a.exec();
}
