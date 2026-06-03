#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <algorithm>
using namespace std;
using namespace cv;

Mat_<uchar> gamma_correction(Mat_<uchar> img, float gamma);

void lab1(){
    Mat_<Vec3b> img(300,200);
    img.setTo(255);

    for (int j = 0; j < img.cols; j++){
        img(img.rows / 2, j) = {0, 0, 255}; //blue, green, red
    }
    imshow("my image", img);

    waitKey();
}
vector<Mat_<uchar>> mysplit(Mat_<Vec3b> img) {
    Mat_<uchar> red(img.rows,img.cols), green(img.rows,img.cols), blue(img.rows,img.cols);
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            blue(i,j)=img(i,j)[0];
            green(i,j)=img(i,j)[1];
            red(i,j)=img(i,j)[2];
        }
    }
    return {red, green, blue};
}
void ex1lab2() {
    auto img = imread("Images/flowers_24bits.bmp");
    auto images = mysplit(img);
    imshow("red", images[0]);
    imshow("green", images[1]);
    imshow("blue", images[2]);
    waitKey();
}
Mat_<uchar> convertRGBtoGray(Mat_<Vec3b> img) {
    Mat_<uchar> grayImg(img.rows, img.cols);
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            uchar R = img(i,j)[2];
            uchar G = img(i,j)[1];
            uchar B = img(i,j)[0];
            grayImg(i,j) = (R + G + B) / 3;
        }
    }
    return grayImg;
}
void ex2lab2() {
    auto img = imread("Images/flowers_24bits.bmp");
    auto image = convertRGBtoGray(img);
    imshow("gray image", image);
    waitKey();
}
Mat_<uchar> convertGrayToBinary(Mat_<uchar> grayImg, uchar threshold) {
    Mat_<uchar> binaryImg(grayImg.rows, grayImg.cols);
    for (int i=0; i<grayImg.rows; i++) {
        for (int j=0; j<grayImg.cols; j++) {
            if (grayImg(i,j)>=threshold)
                binaryImg(i,j)=255;
            else
                binaryImg(i,j) = 0;
        }
    }
    return binaryImg;
}
void ex3lab3() {
    auto img = imread("Images/flowers_24bits.bmp");
    auto grayImg = convertRGBtoGray(img);
    int threshold;
    cout<<"Enter the treshold value: ";
    cin>>threshold;
    auto binaryImg = convertGrayToBinary(grayImg, threshold);
    imshow("binary image", binaryImg);
    waitKey();
}
Mat_<Vec3b> convertRGBtoHSV(Mat_<Vec3b> img) {
    Mat_<Vec3b> hsvImg(img.rows, img.cols);
    for (int i = 0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            float R = img(i,j)[2];
            float G = img(i,j)[1];
            float B = img(i,j)[0];
            float r = R/255.0f;
            float g = G/255.0f;
            float b = B/255.0f;
            float maxVal = max(r, max(g,b));
            float minVal = min(r, min(g,b));
            float c = maxVal-minVal;
            float h, s, v;
            v = maxVal;
            if (v!=0) {
                s=c/v;
            }
            else {
                s=0;
            }
            if (c!=0) {
                if (maxVal==r) {
                    h=60.0f*(((g-b)/c));
                }
                if (maxVal==g) {
                    h=120+60.0f*(((b-r)/c));
                }
                if (maxVal==b) {
                    h=240+60.0f*(((r-g)/c));
                }
            }
            else {
                h=0;
            }
            if (h<0) {
                h=h+360;
            }
            float Hnorm=h*255.0f/360.0f;
            float Snorm=s*255.0f;
            float Vnorm=v*255.0f;
            hsvImg(i,j)={static_cast<uchar>(Hnorm), static_cast<uchar>(Snorm), static_cast<uchar>(Vnorm)};
        }
    }
    return hsvImg;
}
Mat_<Vec3f> convertRGBtoHSV2(Mat_<Vec3b> img) {
    Mat_<Vec3f> hsvImg(img.rows, img.cols);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float R = img(i,j)[2] / 255.0f;
            float G = img(i,j)[1] / 255.0f;
            float B = img(i,j)[0] / 255.0f;
            float maxVal = max(R, max(G, B));
            float minVal = min(R, min(G, B));
            float c = maxVal - minVal;
            float h = 0.0f, s = 0.0f, v = maxVal;
            if (v != 0.0f) {
                s = c / v;
            } else {
                s = 0.0f;
            }
            if (c == 0.0f) {
                h = 0.0f;
            }
            else if (maxVal == R) {
                h = 60.0f * ((G - B) / c);
            }
            else if (maxVal == G) {
                h = 120.0f + 60.0f * ((B - R) / c);
            }
            else {
                h = 240.0f + 60.0f * ((R - G) / c);
            }
            if (h < 0.0f) {
                h += 360.0f;
            }
            hsvImg(i,j) = {h, s, v};
        }
    }
    return hsvImg;
}
Mat_<Vec3b> reconstructHSV(Mat_<float> H, Mat_<float> S, Mat_<float> V) {
    Mat_<Vec3b> hsvImg(H.rows, H.cols);
    for (int i = 0; i < H.rows; i++) {
        for (int j = 0; j < H.cols; j++) {
            hsvImg(i, j)[0] = H(i, j);
            hsvImg(i, j)[1] = S(i, j);
            hsvImg(i, j)[2] = V(i, j);
        }
    }
    return hsvImg;
}
void ex4lab2() {
    auto img = imread("Images/flowers_24bits.bmp");
    auto hsvImg = convertRGBtoHSV(img);
    auto imagesHSV = mysplit(hsvImg);
    //Mat_<Vec3b> hsvImgRec = reconstructHSV(imagesHSV[0], imagesHSV[1], imagesHSV[2]);
    //imshow("HSV image", hsvImg);
    imshow("H", imagesHSV[0]);
    imshow("S", imagesHSV[1]);
    imshow("V", imagesHSV[2]);


    Mat bgr1;
    //cvtColor(img2, hsv, COLOR_BGR2HSV);
    cvtColor(hsvImg, bgr1, COLOR_HSV2BGR);
    imshow("BGR1", bgr1);

    Mat bgr2, hsvImg2;
    auto img2 = imread("Images/flowers_24bits.bmp");
    cvtColor(img2, hsvImg2, COLOR_BGR2HSV);
    cvtColor(hsvImg2, bgr2, COLOR_HSV2BGR);
    imshow("BGR2", bgr2);

    Mat bgr3;
    auto img3 = imread("Images/flowers_24bits.bmp");
    auto hsvImg3 = convertRGBtoHSV2(img);
    cvtColor(hsvImg3, bgr3, COLOR_HSV2BGR);
    imshow("BGR3", bgr3);

    waitKey();
}
bool isInside(Mat img, int i, int j) {
    if (i<img.rows&&i>=0&&j<img.cols&&j>=0) {
        return true;
    }
    return false;
}
void ex5lab2() {
    auto img = imread("Images/flowers_24bits.bmp");
    int i, j;
    cout<<"Enter the coordinates (i, j): ";
    cin>>i>>j;
    if (isInside(img, i, j)) {
        cout<<"The initial image has coordinates "<<img.rows<<"x"<<img.cols<<endl;
        cout<<"The pixel is in the image\n";
    }
    else {
        cout<<"The initial image has coordinates "<<img.rows<<"x"<<img.cols<<endl;
        cout<<"The pixel is outside the image\n";
    }
}
void lab2() {
    int op;
    do{
        printf("Menu:\n");
        printf(" 1 - Split image in RGB images \n");
        printf(" 2 - Convert RGB into grayscale \n");
        printf(" 3 - Convert grayscale into binary \n");
        printf(" 4 - Compute H, S, V values from R, G, B \n");
        printf(" 5 - isInside (img, i, j) \n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d",&op);
        switch (op)
        {
            case 1:
                ex1lab2();
                break;
            case 2:
                ex2lab2();
                break;
            case 3:
                ex3lab3();
                break;
            case 4:
                ex4lab2();
                break;
            case 5:
                ex5lab2();
                break;
        }
    }
    while (op!=0);
}

// Helper: draw triangle and fill it using BFS starting from centroid
void drawTriangleAndFill(const Point &p1, const Point &p2, const Point &p3) {
    // image size
    int rows = 480, cols = 640;
    Mat_<Vec3b> img(rows, cols, Vec3b(255,255,255));
    Mat_<uchar> bin(rows, cols, (uchar)255);

    // draw triangle boundary in both images (black)
    line(img, p1, p2, Vec3b(0,0,0), 1);
    line(img, p2, p3, Vec3b(0,0,0), 1);
    line(img, p3, p1, Vec3b(0,0,0), 1);
    line(bin, p1, p2, Scalar(0), 1);
    line(bin, p2, p3, Scalar(0), 1);
    line(bin, p3, p1, Scalar(0), 1);

    // centroid
    Point centroid((p1.x + p2.x + p3.x)/3, (p1.y + p2.y + p3.y)/3);

    // ensure centroid inside image
    if (centroid.x < 0 || centroid.x >= cols || centroid.y < 0 || centroid.y >= rows) {
        centroid = Point(cols/2, rows/2);
    }

    // if centroid lies on boundary, move slightly towards p1 midpoint
    if (bin(centroid.y, centroid.x) == 0) {
        centroid.x = (centroid.x + p1.x) / 2;
        centroid.y = (centroid.y + p1.y) / 2;
    }

    // BFS flood fill on bin: fill white (255) region until boundary (0)
    queue<Point> Q;
    if (bin(centroid.y, centroid.x) == 255) {
        Q.push(centroid);
        bin(centroid.y, centroid.x) = 128; // mark filled
    }
    int di[4] = {-1, 0, 1, 0};
    int dj[4] = {0, -1, 0, 1};
    while (!Q.empty()) {
        Point p = Q.front(); Q.pop();
        for (int k=0;k<4;k++) {
            int ny = p.y + di[k];
            int nx = p.x + dj[k];
            if (ny>=0 && ny<rows && nx>=0 && nx<cols) {
                if (bin(ny, nx) == 255) {
                    bin(ny, nx) = 128;
                    Q.push(Point(nx, ny));
                }
            }
        }
    }

    // Paint filled pixels on color image
    for (int i=0;i<rows;i++) {
        for (int j=0;j<cols;j++) {
            if (bin(i,j) == 128) {
                img(i,j) = Vec3b(0,0,255); // red fill
            }
            else if (bin(i,j) == 0) {
                img(i,j) = Vec3b(0,0,0); // boundary
            }
        }
    }

    // Draw centroid marker
    circle(img, centroid, 3, Vec3b(0,255,0), FILLED);

    imshow("Triangle - BFS fill", img);
    waitKey(0);
}

// testPractice: read three points from user (or use defaults) and run draw+fill
void testPractice() {
    int x1,y1,x2,y2,x3,y3;
    cout << "Enter 6 integers for three points (x1 y1 x2 y2 x3 y3), or press Enter to use defaults: ";
    // Try to read a line and parse
    string line;
    getline(cin, line); // consume leftover newline
    getline(cin, line);
    if (line.empty()) {
        // defaults
        x1 = 150; y1 = 50;
        x2 = 100; y2 = 300;
        x3 = 400; y3 = 250;
    } else {
        std::istringstream iss(line);
        if (!(iss >> x1 >> y1 >> x2 >> y2 >> x3 >> y3)) {
            cout << "Invalid input, using defaults." << endl;
            x1 = 150; y1 = 50;
            x2 = 100; y2 = 300;
            x3 = 400; y3 = 250;
        }
    }
    Point p1(x1,y1), p2(x2,y2), p3(x3,y3);
    drawTriangleAndFill(p1,p2,p3);
}

vector<int> calchist(Mat_<uchar> img, int nr_bins = 256)
{
    vector<int> hist(nr_bins);
    for(int i=0; i<img.rows; i++)
    {
        for(int j=0; j<img.cols; j++)
        {
            hist[img(i,j)]++;
        }
    }
    return hist;
}

vector<int> calchistCustom(Mat_<uchar> img, int nr_bins)
{
    vector<int> hist(nr_bins);
    for(int i=0; i<img.rows; i++)
    {
        for(int j=0; j<img.cols; j++)
        {
            hist[img(i,j)*nr_bins/256]++;
        }
    }
    return hist;
}

void showHistogram(const string& name, int* hist, const int hist_cols, const int hist_height) {
    Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
    // constructs a white image
    //computes histogram maximum
    int max_hist = 0;
    for (int i = 0; i<hist_cols; i++)
        if (hist[i] > max_hist)
            max_hist = hist[i];
    double scale = 1.0;
    scale = (double)hist_height / max_hist;
    int baseline = hist_height - 1;
    for (int x = 0; x < hist_cols; x++) {
        Point p1 = Point(x, baseline);
        Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
        line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins
        // colored in magenta
    }
    imshow(name, imgHist);
}

vector<float> pdf(Mat_<uchar> img, int nr_bins = 256)
{
    vector<int> hist = calchist(img, nr_bins);
    vector<float> normalized_histogram(nr_bins);
    int size_matrix = img.rows * img.cols;
    for (int i = 0; i < nr_bins; i++)
    {
        normalized_histogram[i] = (float)hist[i] / size_matrix;
    }
    return normalized_histogram;
}

void showPDF(const string& name, float* hist, const int hist_cols, const int hist_height) {
    Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
    float max_hist = 0;
    for (int i = 0; i<hist_cols; i++)
        if (hist[i] > max_hist)
            max_hist = hist[i];
    double scale = 1.0;
    scale = (double)hist_height / max_hist;
    int baseline = hist_height - 1;
    for (int x = 0; x < hist_cols; x++) {
        Point p1 = Point(x, baseline);
        Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
        line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins
    }
    imshow(name, imgHist);
}
vector<int> multilevelThresholding(Mat_<uchar> img, int nr_bins=256, int wh=5, float threshold=0.0003)
{
    vector<float> normalizedHistogram=pdf(img);
    vector <int> local;
    local.push_back(0);
    for(int k = wh; k<normalizedHistogram.size()-wh; k++)
    {
        //k-wh, k+wh
        float v=0;
        bool isLocalMax = true;
        for(int i = k-wh; i<=k+wh; i++)
        {
            v+=normalizedHistogram[i];
            if (normalizedHistogram[i] > normalizedHistogram[k])
                isLocalMax = false;
        }
        v/=(float)wh*2+1;
        if(normalizedHistogram[k]>v+threshold && isLocalMax)
        {
            local.push_back(k);
        }
    }
    local.push_back(255);
    return local;
}
int findClosestMax(int val, vector<int>&maxima)
{
    int best=maxima[0];
    int minDist=abs(val-best);
    for(int i=1; i<maxima.size(); i++)
    {
        int dist=abs(val-maxima[i]);
        if(dist<minDist)
        {
            minDist=dist;
            best=maxima[i];
        }
    }
    return best;
}
Mat_<uchar> applyThreshold(Mat_<uchar> img, vector<int>& maxima)
{
    Mat_<uchar> result(img.rows, img.cols);
    for(int i=0; i<img.rows; i++)
    {
        for(int j=0; j<img.cols; j++)
        {
            result(i,j)=findClosestMax(img(i,j),maxima);
        }
    }
    return result;
}
int saturate(int val)
{
    if(val>255)
        return 255;
    if(val<0)
        return 0;
    return val;
}
void FloydSteinberg (Mat_<uchar> &img, int nr_bins=256)
{
    vector<int> maxima=multilevelThresholding(img);
    for(int i=0; i<img.rows; i++)
    {
        for(int j=0; j<img.cols; j++)
        {
            int oldPixel=img(i,j);
            int newPixel=findClosestMax(oldPixel,maxima);
            img(i,j)=newPixel;
            int error=oldPixel-newPixel;
            if(isInside(img, i, j+1))
            {
                img(i,j+1)=saturate(img(i,j+1)+7*error/16);
            }
            if(isInside(img, i+1, j-1))
            {
                img(i+1,j-1)=saturate(img(i+1,j-1)+3*error/16);
            }
            if(isInside(img, i+1, j))
            {
                img(i+1,j)=saturate(img(i+1,j)+5*error/16);
            }
            if(isInside(img, i+1, j+1))
            {
                img(i+1,j+1)=saturate(img(i+1,j+1)+error/16);
            }
        }
    }
}
void HSVMultilevelThresholding()
{
    Mat_<Vec3b> img = imread("Images/flowers_24bits.bmp");
    Mat_<Vec3b> hsvImg = convertRGBtoHSV(img);
    auto channels = mysplit(hsvImg);
    Mat_<uchar> H = channels[0];
    Mat_<uchar> S = channels[1];
    Mat_<uchar> V = channels[2];
    vector<int> maxima = multilevelThresholding(H);
    for(int i = 0; i < H.rows; i++)
    {
        for(int j = 0; j < H.cols; j++)
        {
            H(i,j) = findClosestMax(H(i,j), maxima);
        }
    }
    Mat_<Vec3b> hsvResult(hsvImg.rows, hsvImg.cols);
    for(int i = 0; i < hsvImg.rows; i++)
    {
        for(int j = 0; j < hsvImg.cols; j++)
        {
            hsvResult(i,j)[0] = H(i,j);
            hsvResult(i,j)[1] = S(i,j);
            hsvResult(i,j)[2] = V(i,j);
        }
    }
    imshow("Original", img);
    imshow("Thresholded Hue", H);
    waitKey(0);
}
void task7(const string& name, int* hist, const int hist_cols, const int hist_height) {
    Mat imgHist(hist_cols, hist_height, CV_8UC3, CV_RGB(170, 255, 255));
    //histogram maximum
    int max_hist = 0;
    for (int i = 0; i<hist_cols; i++)
        if (hist[i] > max_hist)
            max_hist = hist[i];
    double scale = 1.0;
    scale = (double)hist_cols / max_hist;
    for (int y = 0; y < hist_cols; y++) {
        Point p1 = Point(0, y);  //start from left
        Point p2 = Point(cvRound(hist[y])*scale, y); //go to right
        line(imgHist, p1, p2, CV_RGB(130, 27, 100)); // histogram bins
    }
    imshow(name, imgHist);
}
void lab3() {
    int op;
    do {
        printf("Menu:\n");
        printf(" 1 - Histogram \n");
        printf(" 2 - Normalised Histogram \n");
        printf(" 3 - Histogram for given nr of bits \n");
        printf(" 4 - Multilevel Thresholding \n");
        printf(" 5 - Floyd-Steinberg \n");
        printf(" 6 - HSV \n");
        printf(" 7 - Histogram turned upside down \n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d", &op);

        switch (op)
        {
            case 1:
            {
                Mat_<uchar> img = imread("Images/cameraman.bmp", 0);
                auto h = calchist(img);
                showHistogram("hist", h.data(), (int)h.size(), 300);
                waitKey(0);
                break;
            }
            case 2:
            {
                Mat_<uchar> img2 = imread("Images/cameraman.bmp", 0);
                auto h2 = pdf(img2);
                showPDF("pdf", h2.data(), (int)h2.size(), 300);
                waitKey(0);
                break;
            }

            case 3:
            {
                Mat_<uchar> img = imread("Images/cameraman.bmp", 0);
                auto h = calchistCustom(img,130);
                showHistogram("hist", h.data(), (int)h.size(), 300);
                waitKey(0);
                break;
            }

            case 4:
            {
                Mat_<uchar> img2 = imread("Images/cameraman.bmp", 0);
                vector<int> maxima=multilevelThresholding(img2);
                Mat_<uchar> result = applyThreshold(img2, maxima);
                showHistogram("hist", maxima.data(), (int)maxima.size(), 300);
                imshow("Original", img2);
                imshow("Multilevel Thresholding", result);
                waitKey(0);
                break;
            }
            case 5:
            {
                Mat_<uchar> img2 = imread("Images/saturn.bmp", 0);
                FloydSteinberg(img2);
                imshow("FloydSteinberg", img2);
                waitKey(0);
                break;
            }
            case 6:
            {
                HSVMultilevelThresholding();
                break;
            }
            case 7:
            {
                Mat_<uchar> img = imread("Images/cameraman.bmp", 0);
                auto h = calchist(img);
                task7("hist", h.data(), (int)h.size(), 300);
                waitKey(0);
                break;
            }
        }
    } while (op != 0);
}

//1 area -> print value
//2 center of mass -> draw a cross centered at it
//3 axis of elongation -> draw a line with found angle passing through the center of mass
//4 perimeter -> color pixels from it with green
//5 thinness ratio -> print value
//6 aspect ratio -> print value and draw bounding box
//7 projection -> draw in different image
void drawLine(Mat_<Vec3b>& img, Point p1, Point p2, Vec3b color, int thickness = 1) {
    line(img, p1, p2, color, thickness);
}
int areaCalc(Mat_<Vec3b> img, Vec3b color) {
    int area=0;
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            Vec3b pixel=img(i,j);
            if (pixel==color) {
                area=area+1;
            }
        }
    }
    return area;
}
Point centerOfMass(Mat_<Vec3b> img, Vec3b color) {
    int sumX=0, sumY=0, count=0;
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            Vec3b pixel=img(i,j);
            if (pixel==color) {
                sumX=sumX+j;
                sumY=sumY+i;
                count=count+1;
            }
        }
    }
    return Point(sumX/count, sumY/count);
}
float elongationAxis(Mat_<Vec3b> img, Vec3b color) {
    float sum_denominator=0, sum_numerator=0;
    Point center = centerOfMass(img, color);
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            Vec3b pixel=img(i,j);
            if (pixel==color) {
                sum_denominator += (j - center.x)*(j - center.x) - (i - center.y)*(i - center.y);
                sum_numerator=sum_numerator+(i-center.y)*(j-center.x);
            }
        }
    }
    sum_numerator=sum_numerator*2;
    float angle = 0.5 * atan2(sum_numerator, sum_denominator);
    return angle;
}
Mat_<Vec3b> perimeter(Mat_<Vec3b> img, Vec3b color, int &perimeter) {
    Mat_<Vec3b> result = img.clone();
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            Vec3b pixel=img(i,j);
            if (pixel==color) {
                if (isInside(img, i-1, j) && img(i-1,j)!=color ||
                    isInside(img, i+1, j) && img(i+1,j)!=color ||
                    isInside(img, i, j-1) && img(i,j-1)!=color ||
                    isInside(img, i, j+1) && img(i,j+1)!=color) {
                    result(i,j)={0,255,0};
                    perimeter=perimeter+1;
                }
            }
        }
    }
    return result;
}
float thinnessRatio(Mat_<Vec3b> img, Vec3b color) {
    int area = areaCalc(img, color), perim=0;
    Mat_<Vec3b> perimImage = perimeter(img, color, perim);
    return 4*CV_PI*area/(perim*perim);
}
vector<Point> aspectRatio(Mat_<Vec3b> img, Vec3b color, float &ratio) {
    vector<Point> points;
    int minX=img.cols, maxX=0, minY=img.rows, maxY=0;
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            Vec3b pixel=img(i,j);
            if (pixel==color) {
                if (j<minX) minX=j;
                if (j>maxX) maxX=j;
                if (i<minY) minY=i;
                if (i>maxY) maxY=i;
            }
        }
    }
    ratio = (float)(maxX-minX)/(maxY-minY);
    return {Point(minX, minY), Point(maxX, maxY), Point(minX, maxY), Point(maxX, minY)};
}
Mat_<Vec3b> horizontalProjectionImage(Mat_<Vec3b>& img, Vec3b color) {
    Mat_<Vec3b> result(img.rows, img.cols, Vec3b(255, 255, 255));
    vector<int> horizontalProjection(img.rows, 0);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == color) {
                horizontalProjection[i]++;
            }
        }
    }
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < horizontalProjection[i]; j++) {
            result(i, j) = Vec3b(255, 0, 0);
        }
    }
    return result;
}
Mat_<Vec3b> verticalProjectionImage(Mat_<Vec3b>& img, Vec3b color) {
    Mat_<Vec3b> result(img.rows, img.cols, Vec3b(255, 255, 255));
    vector<int> verticalProjection(img.cols, 0);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == color) {
                verticalProjection[j]++;
            }
        }
    }
    for (int j = 0; j < img.cols; j++) {
        for (int i = img.rows - 1; i >= img.rows - verticalProjection[j]; i--) {
            result(i, j) = Vec3b(0, 255, 0);
        }
    }
    return result;
}
Mat_<Vec3b> combinedProjectionImage(Mat_<Vec3b>& img, Vec3b color) {
    vector<int> horizontalProjection(img.rows, 0);
    vector<int> verticalProjection(img.cols, 0);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == color) {
                horizontalProjection[i]++;
                verticalProjection[j]++;
            }
        }
    }
    Mat_<Vec3b> result(2 * img.rows, img.cols, Vec3b(255, 255, 255));
    for (int j = 0; j < img.cols; j++) {
        for (int i = img.rows - 1; i >= img.rows - verticalProjection[j]; i--) {
            result(i, j) = Vec3b(0, 255, 0);
        }
    }
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < horizontalProjection[i]; j++) {
            result(i + img.rows, j) = Vec3b(255, 0, 0);
        }
    }
    return result;
}
void lab4Menu(Mat_<Vec3b> img, Vec3b color) {
    int op;
    do {
        printf("Menu:\n");
        printf("1 - Area \n");
        printf("2 - Center of mass \n");
        printf("3 - Axis of elongation \n");
        printf("4 - Perimeter \n");
        printf("5 - Thinness ratio \n");
        printf("6 - Aspect ratio \n");
        printf("7 - Projection \n");
        printf("0 - Exit\n\n");
        printf("Option: ");
        scanf("%d", &op);
        switch (op) {
            case 1:
            {
                cout<<areaCalc(img, color)<<endl;
                break;
            }
            case 2: {
                Point center = centerOfMass(img, color);
                Mat_<Vec3b> imgCopy = img.clone();
                drawLine(imgCopy, Point(center.x - 10, center.y), Point(center.x + 10, center.y), Vec3b(0, 0, 0), 2);
                drawLine(imgCopy, Point(center.x, center.y - 10), Point(center.x, center.y + 10), Vec3b(0, 0, 0), 2);
                imshow("center of mass", imgCopy);
                waitKey(0);
                break;
            }
            case 3: {
                Point center = centerOfMass(img, color);
                float angle = elongationAxis(img, color);
                int length = 50;
                Point p1(center.x - length * cos(angle), center.y - length * sin(angle));
                Point p2(center.x + length * cos(angle), center.y + length * sin(angle));
                Mat_<Vec3b> imgCopy = img.clone();
                drawLine(imgCopy, p1, p2, Vec3b(0, 0, 0), 2);
                imshow("axis of elongation", imgCopy);
                waitKey(0);
                break;
            }
            case 4: {
                int perim=0;
                Mat_<Vec3b> perimImage = perimeter(img, color, perim);
                imshow("perimeter", perimImage);
                cout<<perim<<endl;
                waitKey(0);
                break;
            }
            case 5: {
                cout<<thinnessRatio(img, color)<<endl;
                break;
            }
            case 6: {
                float ratio;
                vector<Point> points = aspectRatio(img, color, ratio);
                Mat_<Vec3b> imgCopy = img.clone();
                rectangle(imgCopy, points[0], points[1], Vec3b(0, 0, 255), 4);
                imshow("aspect ratio", imgCopy);
                cout<<ratio<<endl;
                waitKey(0);
                break;
            }
            case 7: {
                Mat_<Vec3b> hProj = horizontalProjectionImage(img, color);
                Mat_<Vec3b> vProj = verticalProjectionImage(img, color);
                Mat_<Vec3b> bothProj = combinedProjectionImage(img, color);
                imshow("Horizontal projection", hProj);
                imshow("Vertical projection", vProj);
                imshow("Both projections", bothProj);
                waitKey(0);
                break;
            }
        }
    } while (op != 0);
}

void onMyMouse(int event, int x, int y, int flags, void* param) {
    if (event==EVENT_LBUTTONDOWN) {
        Mat_<Vec3b> img = *(Mat_<Vec3b> *)param;
        Vec3b color=img(y,x);
        cout<<color<<endl;
        lab4Menu(img, color);
    }

}

void lab4() {
    int op;
    do {
        printf("Menu:\n");
        printf(" 1 - Open menu for calculations \n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d", &op);

        switch (op)
        {
            case 1:
            {
                Mat_<Vec3b> img = imread("PI-L4/trasaturi_geom.bmp");
                imshow("input image",img);
                setMouseCallback("input image", onMyMouse, &img);
                waitKey(0);
                break;
            }
        }
    } while (op != 0);
}

void bfs_traversal(Mat_<uchar> img, Mat_<int> &labels, int i, int j, int label, bool use8Neighbors) {
    queue<Point> Q;
    Q.push(Point(j, i));
    labels(i, j) = label;
    vector<Point> directions;
    if (use8Neighbors) {
        directions.push_back(Point(-1, 0));
        directions.push_back(Point(0, -1));
        directions.push_back(Point(0, 1));
        directions.push_back(Point(1, 0));
        directions.push_back(Point(-1, -1));
        directions.push_back(Point(-1, 1));
        directions.push_back(Point(1, -1));
        directions.push_back(Point(1, 1));
    }
    else {
        directions.push_back(Point(-1, 0));
        directions.push_back(Point(0, -1));
        directions.push_back(Point(0, 1));
        directions.push_back(Point(1, 0));
    }
    while (!Q.empty()) {
        Point pointQ = Q.front();
        Q.pop();
        for (Point dir : directions) {
            Point neighbor = pointQ + dir;
            if (isInside(img, neighbor.y, neighbor.x) && img(neighbor.y, neighbor.x) == 0 && labels(neighbor.y, neighbor.x) == 0) {
                labels(neighbor.y, neighbor.x) = label;
                Q.push(neighbor);
            }
        }
    }
}
void displayLabels(Mat_<int> labels, string windowName) {
    Mat_<Vec3b> colorLabels(labels.rows, labels.cols);
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            int label = labels(i, j);
            if (label == 0) {
                colorLabels(i, j) = Vec3b(255, 255, 255);
            }
            else {
                colorLabels(i, j) = Vec3b((label * 50) % 256, (label * 80) % 256, (label * 110) % 256);
            }
        }
    }
    imshow(windowName, colorLabels);
}

void bfs_connected_components(Mat_<uchar> img, Mat_<int>& labels, bool use8Neighbors) {
    int label = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 0 && labels(i, j) == 0) {
                label++;
                bfs_traversal(img, labels, i, j, label, use8Neighbors);
            }
        }
    }
}
void applyBFS_to_one_image(Mat_<uchar> img) {
    Mat_<int> labels(img.rows, img.cols, 0);
    bool use8Neighbors = false;
    bfs_connected_components(img, labels, use8Neighbors);
    imshow("Initial image", img);
    displayLabels(labels, "4-neighborhood");
    use8Neighbors = true;
    Mat_<int> labels8(img.rows, img.cols, 0);
    bfs_connected_components(img, labels8, use8Neighbors);
    displayLabels(labels8, "8-neighborhood");
    waitKey(0);
}
void applyBFS_to_all_images() {
    Mat_<uchar> img = imread("PI-L5/circle_square.bmp", IMREAD_GRAYSCALE);
    applyBFS_to_one_image(img);
    Mat_<uchar> img2 = imread("PI-L5/crosses.bmp", IMREAD_GRAYSCALE);
    applyBFS_to_one_image(img2);
    Mat_<uchar> img3 = imread("PI-L5/diagonal.bmp", IMREAD_GRAYSCALE);
    applyBFS_to_one_image(img3);
    Mat_<uchar> img4 = imread("PI-L5/disks.bmp", IMREAD_GRAYSCALE);
    applyBFS_to_one_image(img4);
    Mat_<uchar> img5 = imread("PI-L5/letters.bmp", IMREAD_GRAYSCALE);
    applyBFS_to_one_image(img5);
    Mat_<uchar> img6 = imread("PI-L5/shapes.bmp", IMREAD_GRAYSCALE);
    applyBFS_to_one_image(img6);
    Mat_<uchar> img7 = imread("PI-L5/text_binary.bmp", IMREAD_GRAYSCALE);
    applyBFS_to_one_image(img7);
}

int findSet(vector<int>& parent, int x) {
    if (parent[x] != x)
        parent[x] = findSet(parent, parent[x]); // path compression
    return parent[x];
}

void unite(vector<int>& parent, int a, int b) {
    a = findSet(parent, a);
    b = findSet(parent, b);
    if (a != b)
        parent[b] = a;
}

void twopass_connected_components(Mat_<uchar> img, Mat_<int>& labels, bool use8Neighbors, bool showIntermediate = true) {
    vector<Point> directions;

    vector<int> parent(10000);
    for (int i = 0; i < parent.size(); i++)
        parent[i] = i;

    int label=0;
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            if (img(i,j)==0) {
                vector<int> neighborLabels;
                if (isInside(img, i-1,j)&& labels(i-1,j)>0)
                    neighborLabels.push_back(labels(i-1,j));
                if (isInside(img, i,j-1)&& labels(i,j-1)>0)
                    neighborLabels.push_back(labels(i,j-1));
                if (use8Neighbors) {
                    if (isInside(img, i-1,j-1)&& labels(i-1,j-1)>0)
                        neighborLabels.push_back(labels(i-1,j-1));
                    if (isInside(img, i-1,j+1)&& labels(i-1,j+ 1)>0)
                        neighborLabels.push_back(labels(i-1,j+1));
                }
                if (neighborLabels.empty()) {
                    label++;
                    labels(i,j)=label;
                }
                else {
                    int minLabel = neighborLabels[0];
                    for (int l : neighborLabels)
                        if (l < minLabel)
                            minLabel = l;
                    labels(i,j) = minLabel;
                    for (int l : neighborLabels) {
                        if (l != minLabel) {
                            unite(parent, minLabel, l);
                        }
                    }
                }
            }
        }
    }

    if (showIntermediate) {
        displayLabels(labels, "between passes");
    }

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            if (labels(i,j) > 0) {
                labels(i,j) = findSet(parent, labels(i,j));
            }
        }
    }
}
void apply2pass_to_one_image(Mat_<uchar> img) {
    Mat_<int> labels(img.rows, img.cols, 0);
    bool use8Neighbors = false;
    twopass_connected_components(img, labels, use8Neighbors);
    imshow("Initial image", img);
    displayLabels(labels, "4-neighborhood");
    waitKey(0);
    use8Neighbors = true;
    Mat_<int> labels8(img.rows, img.cols, 0);
    twopass_connected_components(img, labels8, use8Neighbors);
    displayLabels(labels8, "8-neighborhood");
    waitKey(0);
}
void apply2pass_to_all_images() {
    Mat_<uchar> img = imread("PI-L5/circle_square.bmp", IMREAD_GRAYSCALE);
    apply2pass_to_one_image(img);
    Mat_<uchar> img2 = imread("PI-L5/crosses.bmp", IMREAD_GRAYSCALE);
    apply2pass_to_one_image(img2);
    Mat_<uchar> img3 = imread("PI-L5/diagonal.bmp", IMREAD_GRAYSCALE);
    apply2pass_to_one_image(img3);
    Mat_<uchar> img4 = imread("PI-L5/disks.bmp", IMREAD_GRAYSCALE);
    apply2pass_to_one_image(img4);
    Mat_<uchar> img5 = imread("PI-L5/letters.bmp", IMREAD_GRAYSCALE);
    apply2pass_to_one_image(img5);
    Mat_<uchar> img6 = imread("PI-L5/shapes.bmp", IMREAD_GRAYSCALE);
    apply2pass_to_one_image(img6);
    Mat_<uchar> img7 = imread("PI-L5/text_binary.bmp", IMREAD_GRAYSCALE);
    apply2pass_to_one_image(img7);
}

void dfs_traversal(Mat_<uchar> img, Mat_<int> &labels, int i, int j, int label, bool use8Neighbors) {
    stack<Point> S;
    S.push(Point(j, i));
    labels(i, j) = label;
    vector<Point> directions;
    if (use8Neighbors) {
        directions.push_back(Point(-1, 0));
        directions.push_back(Point(0, -1));
        directions.push_back(Point(0, 1));
        directions.push_back(Point(1, 0));
        directions.push_back(Point(-1, -1));
        directions.push_back(Point(-1, 1));
        directions.push_back(Point(1, -1));
        directions.push_back(Point(1, 1));
    }
    else {
        directions.push_back(Point(-1, 0));
        directions.push_back(Point(0, -1));
        directions.push_back(Point(0, 1));
        directions.push_back(Point(1, 0));
    }
    while (!S.empty()) {
        Point current = S.top();
        S.pop();
        for (Point dir : directions) {
            Point neighbor = current + dir;
            if (isInside(img, neighbor.y, neighbor.x) && img(neighbor.y, neighbor.x) == 0 && labels(neighbor.y, neighbor.x) == 0) {
                labels(neighbor.y, neighbor.x) = label;
                S.push(neighbor);
            }
        }
    }
}
void dfs_connected_components(Mat_<uchar> img, Mat_<int> labels, bool use8Neighbors) {
    int label = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 0 && labels(i, j) == 0) {
                label++;
                dfs_traversal(img, labels, i, j, label, use8Neighbors);
            }
        }
    }
}
void applyDFS_to_one_image(Mat_<uchar> img) {
    Mat_<int> labels(img.rows, img.cols, 0);
    bool use8Neighbors = false;
    dfs_connected_components(img, labels, use8Neighbors);
    imshow("Initial image", img);
    displayLabels(labels, "4-neighborhood");
    use8Neighbors = true;
    Mat_<int> labels8(img.rows, img.cols, 0);
    dfs_connected_components(img, labels8, use8Neighbors);
    displayLabels(labels8, "8-neighborhood");
    waitKey(0);
}
void applyDFS_to_all_images() {
    Mat_<uchar> img = imread("PI-L5/circle_square.bmp", IMREAD_GRAYSCALE);
    applyDFS_to_one_image(img);
    Mat_<uchar> img2 = imread("PI-L5/crosses.bmp", IMREAD_GRAYSCALE);
    applyDFS_to_one_image(img2);
    Mat_<uchar> img3 = imread("PI-L5/diagonal.bmp", IMREAD_GRAYSCALE);
    applyDFS_to_one_image(img3);
    Mat_<uchar> img4 = imread("PI-L5/disks.bmp", IMREAD_GRAYSCALE);
    applyDFS_to_one_image(img4);
    Mat_<uchar> img5 = imread("PI-L5/letters.bmp", IMREAD_GRAYSCALE);
    applyDFS_to_one_image(img5);
    Mat_<uchar> img6 = imread("PI-L5/shapes.bmp", IMREAD_GRAYSCALE);
    applyDFS_to_one_image(img6);
    Mat_<uchar> img7 = imread("PI-L5/text_binary.bmp", IMREAD_GRAYSCALE);
    applyDFS_to_one_image(img7);
}
void lab5() {
    int op;
    do{
        printf("Menu:\n");
        printf(" 1 - BFS Traversal on all Images \n");
        printf(" 2 - 2pass Traversal on all Images \n");
        printf(" 3 - DFS Traversal on all Images \n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d",&op);
        switch (op)
        {
            case 1:
                applyBFS_to_all_images();
                break;
            case 2:
                apply2pass_to_all_images();
                break;
            case 3:
                applyDFS_to_all_images();
        }
    }
    while (op!=0);
}

void border_tracing(Mat_<uchar> img, vector<Point>& border, vector<pair<int,int>>& directions) {
    int di[] = {0, -1, -1, -1, 0, 1, 1, 1};
    int dj[] = {1, 1, 0, -1, -1, -1, 0, 1};
    border.clear();
    directions.clear();
    Point start(-1, -1);
    bool found = false;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 0) {
                start = Point(j, i);
                found = true;
                break;
            }
        }
        if (found) {
            break;
        }
    }
    if (!found) {
        return;
    }

    Point current = start;
    Point p1(-1, -1);
    int dir = 7;
    border.push_back(start);

    while (true) {
        int startDir;
        if(dir % 2 == 0) {
            startDir = (dir + 7) % 8;
        }
        else {
            startDir = (dir + 6) % 8;
        }
        bool foundNext = false;
        Point next;
        int nextDir = dir;
        for (int k = 0; k < 8; k++) {
            int neighbourDir = (startDir + k) % 8;
            Point neighbor(current.x + dj[neighbourDir], current.y + di[neighbourDir]);
            if (isInside(img, neighbor.y, neighbor.x) && img(neighbor.y, neighbor.x) == img(current.y, current.x)) {
                next = neighbor;
                nextDir = neighbourDir;
                foundNext = true;
                break;
            }
        }
        if (!foundNext) {
            break;
        }
        if (p1.x!=-1 && current==start && next==p1) {
            break;
        }
        border.push_back(next);
        directions.push_back({di[nextDir], dj[nextDir]});
        if (p1.x == -1) {
            p1 = next;
        }
        current = next;
        dir = nextDir;
    }
    if (border.size() > 1 && border.back() == start) {
        border.pop_back();
        if (!directions.empty()) {
            directions.pop_back();
        }
    }
}

void call_border_tracing() {
    Mat_<uchar> img = imread("PI-L6/triangle_up.bmp", IMREAD_GRAYSCALE);
    vector<Point> border;
    vector<pair<int,int>> directions;
    border_tracing(img, border, directions);
    Mat_<Vec3b> borderImage(img.rows, img.cols, Vec3b(255, 255, 255));
    for (Point p : border) {
        borderImage(p.y, p.x) = Vec3b(0, 0, 255);
    }
    imshow("Initial image", img);
    imshow("Border tracing", borderImage);
    waitKey(0);
}

vector<int> chain_code(Mat_<uchar>img) {
    int di[] = {0, -1, -1, -1, 0, 1, 1, 1};
    int dj[] = {1, 1, 0, -1, -1, -1, 0, 1};
    vector<Point> border;
    vector<pair<int,int>> directions;
    border_tracing(img, border, directions);
    vector<int> chainCode;
    for (pair<int,int> currentDir: directions) {
           for (int k=0; k<8; k++) {
               if (currentDir.first==di[k]&&currentDir.second==dj[k]) {
                   chainCode.push_back(k);
                   break;
               }
           }
    }
    return chainCode;
}

vector<int> derivative_code(Mat_<uchar>img) {
    vector<int> derivative;
    vector<int> chainCode = chain_code(img);
    for (int i=0; i<chainCode.size(); i++) {
        int diff = (chainCode[(i+1)%chainCode.size()] - chainCode[i] + 8) % 8;
        derivative.push_back(diff);
    }
    return derivative;
}

void print_vector_to_file(const vector<int>& vec, const string& filename) {
    ofstream fout(filename);
    for (int num : vec) {
        fout<<num<<" ";
    }
    fout.close();
}

void call_chainCode_derivativeCode() {
    Mat_<uchar> img = imread("PI-L6/triangle_up.bmp", IMREAD_GRAYSCALE);
    vector<int> chainCode = chain_code(img);
    print_vector_to_file(chainCode, "chain_code.txt");
    vector<int> derivative = derivative_code(img);
    print_vector_to_file(derivative, "derivative_code.txt");
}

void reconstruct_image(const vector<int>& chainCode, Point start) {
    Mat_<Vec3b> img=imread("PI-L6/gray_background.bmp");
    int di[] = {0, -1, -1, -1, 0, 1, 1, 1};
    int dj[] = {1, 1, 0, -1, -1, -1, 0, 1};
    Point current = start;
    for (int dir : chainCode) {
        current.y += di[dir];
        current.x += dj[dir];
        if (isInside(img, current.y, current.x)) {
            img(current.y, current.x) = Vec3b(0, 0, 255);
        }
    }
    imshow("Reconstructed image", img);
    waitKey(0);
}

void call_reconstruct_image() {
    vector<int> chainCode;
    ifstream fin("PI-L6/reconstruct.txt");
    Point start;
    fin >> start.x >> start.y;
    int expectedCount;
    fin >> expectedCount;
    int num;
    while (fin >> num) {
        chainCode.push_back(num);
    }
    fin.close();
    reconstruct_image(chainCode, start);
}

void lab6() {
    int op;
    do{
        printf("Menu:\n");
        printf(" 1 - Border tracing \n");
        printf(" 2 - Chain Code and Derivative Code \n");
        printf(" 3 - Reconstruct image from Chain Code \n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d",&op);
        switch (op)
        {
            case 1:
                call_border_tracing();
                break;
            case 2:
                call_chainCode_derivativeCode();
                break;
            case 3:
                call_reconstruct_image();
                break;
        }
    }
    while (op!=0);
}

Mat_<uchar> dilation(Mat_<uchar> src, Mat_<uchar> strel) {
    Mat_<uchar> dst(src.size());
    dst.setTo(255);
    for (int i=0; i<src.rows; i++) {
        for (int j=0; j<src.cols; j++) {
            if (src(i,j)!=0) {
                continue;
            }
            for (int u=0; u<strel.rows; u++) {
                for (int v=0; v<strel.cols; v++) {
                    if (strel(u,v)==0) {
                        int i2 = i+u-strel.rows/2;
                        int j2 = j+v-strel.cols/2;
                        if (isInside(src, i2, j2)) {
                            dst(i2,j2)=0;
                        }
                    }
                }
            }
        }
    }
    return dst;
}
Mat_<uchar> erotion(Mat_<uchar> src, Mat_<uchar> strel) {
    Mat_<uchar> dst(src.size());
    dst.setTo(255);
    for (int i=0; i<src.rows; i++) {
        for (int j=0; j<src.cols; j++) {
            bool coveredByStructuringElement = true;
            for (int u =0; u<strel.rows; u++) {
                for (int v=0; v<strel.cols; v++) {
                    if (strel(u,v)==0) {
                        int i2 = i+u-strel.rows/2;
                        int j2 = j+v-strel.cols/2;
                        if (!isInside(src, i2, j2) || src(i2,j2)!=0) {
                            coveredByStructuringElement=false;
                            break;
                        }
                    }
                }
                if (!coveredByStructuringElement) {
                    break;
                }
            }
            if (coveredByStructuringElement) {
                dst(i,j)=0;
            }
        }
    }
    return dst;
}

vector<pair<string, Mat_<uchar>>> getHardcodedStrels() {
    vector<pair<string, Mat_<uchar>>> strels;

    Mat_<uchar> square3(3, 3);
    square3.setTo(0);
    strels.push_back({"Square 3x3", square3});

    Mat_<uchar> cross3(3, 3);
    cross3.setTo(255);
    cross3(0, 1) = 0;
    cross3(1, 0) = 0;
    cross3(1, 1) = 0;
    cross3(1, 2) = 0;
    cross3(2, 1) = 0;
    strels.push_back({"Cross 3x3", cross3});

    Mat_<uchar> diamond5(5, 5);
    diamond5.setTo(255);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (abs(i - 2) + abs(j - 2) <= 2) {
                diamond5(i, j) = 0;
            }
        }
    }
    strels.push_back({"Diamond 5x5", diamond5});

    Mat_<uchar> diamond57(5, 7);
    diamond57.setTo(255);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 7; j++) {
            if (abs(i - 2) + abs(j - 3) <= 2) {
                diamond57(i, j) = 0;
            }
        }
    }
    strels.push_back({"Diamond 5x7", diamond57});

    Mat_<uchar> lineH(1, 7);
    lineH.setTo(0);
    strels.push_back({"Horizontal line 1x7", lineH});

    Mat_<uchar> lineV(7, 1);
    lineV.setTo(0);
    strels.push_back({"Vertical line 7x1", lineV});

    Mat_<uchar> rect26(2, 6);
    rect26.setTo(0);
    strels.push_back({"Rectangle 2x6", rect26});

    return strels;
}

Mat_<uchar> buildLargeStrelPreview(const Mat_<uchar>& strel) {
    // Display-only scaling so tiny kernels (1x7, 3x3) are visible in imshow.
    const int targetSize = 240;
    int maxDim = max(strel.rows, strel.cols);
    int scale = max(1, targetSize / maxDim);
    Mat_<uchar> preview;
    resize(strel, preview, Size(strel.cols * scale, strel.rows * scale), 0, 0, INTER_NEAREST);
    return preview;
}

void lab7() {
    int op;
    do{
        printf("Menu:\n");
        printf(" 1 - Dilation \n");
        printf(" 2 - Erosion \n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d",&op);
        switch (op)
        {
            case 1: {
                Mat_<uchar> src=imread("PI-L5/letters.bmp", IMREAD_GRAYSCALE);
                auto strels = getHardcodedStrels();
                for (const auto& strelData : strels) {
                    Mat_<uchar> dilated = dilation(src, strelData.second);
                    Mat_<uchar> strelPreview = buildLargeStrelPreview(strelData.second);
                    cout << "Dilation with: " << strelData.first << endl;
                    imshow("Original", src);
                    imshow("Strel", strelPreview);
                    imshow("Result", dilated);
                    waitKey(0);
                }
                break;
            }
            case 2: {
                Mat_<uchar> srcE=imread("PI-L7/Morphological_Op_Images/2_Erode/mon1thr1_bw.bmp", IMREAD_GRAYSCALE);
                auto strels = getHardcodedStrels();
                for (const auto& strelData : strels) {
                    Mat_<uchar> eroded = erotion(srcE, strelData.second);
                    Mat_<uchar> strelPreview = buildLargeStrelPreview(strelData.second);
                    cout << "Erosion with: " << strelData.first << endl;
                    imshow("Original", srcE);
                    imshow("Strel", strelPreview);
                    imshow("Result", eroded);
                    waitKey(0);
                }
                break;
            }
        }
    }
    while (op!=0);
}

double mean(Mat_<uchar> img) {
    vector<int> hist = calchist(img, 256);
    double M = (double)img.rows * img.cols;
    double mean = 0;
    for (int i = 0; i < hist.size(); i++) {
        mean += i * hist[i] / M;
    }
    return mean;
}

double standard_deviation(Mat_<uchar> img) {
    double meanVar = mean(img);
    vector<int> hist = calchist(img, 256);
    double M = (double)img.rows * img.cols;
    double var = 0;
    for (int i = 0; i < hist.size(); i++) {
        var += hist[i] * pow(i - meanVar, 2);
    }
    var /= M;
    return sqrt(var);
}

vector<int> cumulative_histogram(Mat_<uchar> img) {
    vector<int> hist = calchist(img, 256);
    vector<int> cumHist(hist.size(), 0);
    cumHist[0] = hist[0];
    for (int i = 1; i < hist.size(); i++) {
        cumHist[i] = cumHist[i-1] + hist[i];
    }
    return cumHist;
}

void ex1lab8() {
    Mat_<uchar> img = imread("PI-L8/balloons.bmp", IMREAD_GRAYSCALE);
    double meanVal = mean(img);
    double stddevVal = standard_deviation(img);
    cout << "Mean: " << meanVal << endl;
    cout << "Standard Deviation: " << stddevVal << endl;
    vector<int> hist = calchist(img, 256);
    vector<int> cumulativeHist = cumulative_histogram(img);
    showHistogram("Histogram", hist.data(), (int)hist.size(), 300);
    showHistogram("Cumulative Histogram", cumulativeHist.data(), (int)cumulativeHist.size(), 300);
    waitKey(0);
}

float automatic_threshold(Mat_<uchar> img) {
    vector<int> hist = calchist(img, 256);
    int Imin = 0, Imax = 255;
    for (int i = 0; i < hist.size(); i++) {
        if (hist[i] > 0) {
            Imin = i;
            break;
        }
    }
    for (int i = hist.size() - 1; i >= 0; i--) {
        if (hist[i] > 0) {
            Imax = i;
            break;
        }
    }
    float T = (Imin + Imax) / 2.0f;
    while (true) {
        float sum1 = 0, sum2 = 0;
        int count1 = 0, count2 = 0;
        for (int i = Imin; i <= Imax; i++) {
            if (i <= T) {
                sum1 += i * hist[i];
                count1 += hist[i];
            } else {
                sum2 += i * hist[i];
                count2 += hist[i];
            }
        }
        if (count1 == 0 || count2 == 0) {
            break;
        }
        float mu1 = sum1 / count1;
        float mu2 = sum2 / count2;
        float new_threshold = (mu1 + mu2) / 2.0f;
        if (abs(new_threshold - T) < 0.1f) {
            T = new_threshold;
            break;
        }
        T = new_threshold;
    }
    return T;
}

vector<int> negative_histogram(const vector<int>& hist) {
    vector<int> negHist(hist.size(), 0);
    for (int i = 0; i < hist.size(); i++) {
        negHist[i] = hist[hist.size() - 1 - i];
    }
    return negHist;
}

vector<int> brightness_histogram(const vector<int>& hist) {
    vector<int> brightHist(hist.size(), 0);
    for (int i = 0; i < hist.size(); i++) {
        brightHist[i] = hist[i];
    }
    return brightHist;
}

vector<int> stretch_shrink_histogram(vector<int>& hist, int g_out_min, int g_out_max, Mat_<uchar> img, Mat_<uchar>*imgNew) {
    vector<int> stretchHist(hist.size(), 0);
    int g_in_min = 0, g_in_max = 255;
    for (int i = 0; i < hist.size(); i++) {
        if (hist[i] > 0) {
            g_in_min = i;
            break;
        }
    }
    for (int i = hist.size() - 1; i >= 0; i--) {
        if (hist[i] > 0) {
            g_in_max = i;
            break;
        }
    }
    // for (int i = g_in_min; i <= g_in_max; i++) {
    //     int g_out = g_out_min + (i - g_in_min) * (float)(g_out_max - g_out_min) / (g_in_max - g_in_min);
    //     if (g_out<0) {
    //         g_out=0;
    //     }
    //     if (g_out>255) {
    //         g_out=255;
    //     }
    //     stretchHist[g_out] += hist[i];
    // }
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int g_in = img(i, j);
            int g_out = g_out_min + (g_in - g_in_min) * (float)(g_out_max - g_out_min) / (g_in_max - g_in_min);
            if (g_out < 0) {
                g_out = 0;
            }
            if (g_out > 255) {
                g_out = 255;
            }
            (*imgNew)(i, j) = g_out;
            stretchHist[g_out]++;
        }
    }
    return stretchHist;
}

Mat_<uchar> gamma_correction(Mat_<uchar> img, float gamma) {
    Mat_<uchar> corrected(img.size());
    float L = 255.0f;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float normalized = img(i, j) / L;
            float g_out = pow(normalized, gamma) * L;
            if (g_out < 0) {
                g_out = 0;
            }
            if (g_out > 255) {
                g_out = 255;
            }
            corrected(i, j) = static_cast<uchar>(g_out);
        }
    }
    return corrected;
}

vector<float> cdf(const vector<float>& pdf) {
    vector<float> cdfres(pdf.size(), 0);
    cdfres[0] = pdf[0];
    for (int i = 1; i < pdf.size(); i++) {
        cdfres[i] = cdfres[i-1] + pdf[i];
    }
    return cdfres;
}

void histogram_equalization(Mat_ <uchar> img) {
    vector<float> pdfC = pdf(img);
    vector<float> cdfC = cdf(pdfC);
    float L=255.0f;
    vector<int> equalizedHist(pdfC.size(), 0);
    Mat_<uchar> equalizedImg(img.size());

    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            int g_in = img(i, j);
            int g_out = static_cast<int>(cdfC[g_in] * L);
            if (g_out < 0) {
                g_out = 0;
            }
            if (g_out > 255) {
                g_out = 255;
            }
            equalizedImg(i, j) = g_out;
            equalizedHist[g_out]++;
        }
    }
    imshow("Original Image", img);
    imshow("Equalized Image", equalizedImg);
    showHistogram("Equalized Histogram", equalizedHist.data(), (int)equalizedHist.size(), 300);
}

void lab8() {
    int op;
    do{
        printf("Menu:\n");
        printf(" 1 - Compute and Display the Mean and Standard Deviation, the Histogram and the Cumulative Histograms \n");
        printf(" 2 - Automatic threshold computation and threshold images \n");
        printf(" 3 - Negative histogram \n");
        printf(" 4 - Brightness histogram \n");
        printf(" 5 - Stretch/Shrink histogram \n");
        printf(" 6 - Gamma correction \n");
        printf(" 7- Histogram equalization \n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d",&op);
        switch (op)
        {
            case 1: {
                ex1lab8();
                break;
            }
            case 2: {
                Mat_<uchar> img = imread("Images/eight.bmp", IMREAD_GRAYSCALE);
                int T = automatic_threshold(img);
                cout << "Automatic Threshold: " << T << endl;
                Mat_<uchar> thresholded=convertGrayToBinary(img, T);
                imshow("Original Image", img);
                imshow("Thresholded Image", thresholded);
                waitKey(0);
                break;
            }
            case 3: {
                Mat_<uchar> img = imread("Images/eight.bmp", IMREAD_GRAYSCALE);
                vector<int> hist = calchist(img, 256);
                vector<int> negHist = negative_histogram(hist);
                showHistogram("Negative Histogram", negHist.data(), (int)negHist.size(), 300);
                waitKey(0);
                break;
            }
            case 4: {
                Mat_<uchar> img = imread("Images/eight.bmp", IMREAD_GRAYSCALE);
                vector<int> hist = calchist(img, 256);
                vector<int> brightHist = brightness_histogram(hist);
                showHistogram("Brightness Histogram", brightHist.data(), (int)brightHist.size(), 300);
                waitKey(0);
                break;
            }
            case 5: {
                Mat_<uchar> img = imread("PI-L8/Hawkes_Bay_NZ.bmp", IMREAD_GRAYSCALE);
                vector<int> hist = calchist(img, 256);
                Mat_<uchar> imgNew(img.size());
                vector<int> stretchedHist = stretch_shrink_histogram(hist, 10, 250, img, &imgNew);
                showHistogram("Stretched Histogram", stretchedHist.data(), (int)stretchedHist.size(), 300);
                imshow("Stretched Image", imgNew);
                Mat_<uchar> img2 = imread("PI-L8/wheel.bmp", IMREAD_GRAYSCALE);
                vector<int> hist2 = calchist(img2, 256);
                Mat_<uchar> img2New(img2.size());
                vector<int> stretchedHist2 = stretch_shrink_histogram(hist2, 50, 150, img2, &img2New);
                showHistogram("Shrunk Histogram", stretchedHist2.data(), (int)stretchedHist2.size(), 300);
                imshow("Shrunk Image", img2New);
                waitKey(0);
                break;
            }
            case 6: {
                Mat_<uchar> img = imread("PI-L8/wilderness.bmp", IMREAD_GRAYSCALE);
                Mat_<uchar> gammaCorrected = gamma_correction(img, 0.5f);
                imshow("Gamma Encoded", gammaCorrected);
                Mat_<uchar> gammaCorrected2 = gamma_correction(img, 2.0f);
                imshow("Gamma Decoded", gammaCorrected2);
                waitKey(0);
                break;
            }
            case 7: {
                Mat_<uchar> img = imread("PI-L8/Hawkes_Bay_NZ.bmp", IMREAD_GRAYSCALE);
                histogram_equalization(img);
                waitKey(0);
                Mat_<uchar> img2 = imread("PI-L8/wheel.bmp", IMREAD_GRAYSCALE);
                histogram_equalization(img2);
                waitKey(0);
                break;
            }
        }
    }
    while (op!=0);
}

Mat_ <float>convolution(const Mat_<uchar>& img, const Mat_<float>& kernel) {
    int kRows = kernel.rows;
    int kCols = kernel.cols;
    Mat_<float> result_img(img.size());
    result_img.setTo(0);
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            float sum = 0.0f;
            for (int u=0; u<kRows; u++) {
                for (int v=0; v<kCols; v++) {
                    int i2 = i + u - kRows / 2;
                    int j2 = j + v - kCols / 2;
                    if (isInside(img, i2, j2)) {
                        sum += img(i2, j2) * kernel(u, v);
                    }
                }
            }
            result_img(i, j) = sum;
        }
    }
    return result_img;
}

pair<float,float> computeAandB(Mat_<float> img) {
    float negative_sum=0;
    float positive_sum=0;
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            if (img(i, j) < 0) {
                negative_sum += img(i, j);
            }
            else {
                positive_sum+=img(i,j);
            }
        }
    }
    float a = negative_sum*255;
    float b = positive_sum*255;
    cout<<a << " " <<b<<endl;
    return {a, b};
}

Mat_<uchar> applyRawNormalization(Mat_<float> img, Mat_<float> kernel) {
    Mat_<uchar> normalized(img.size());
    pair<float, float> aAndB = computeAandB(kernel);
    float a = aAndB.first;
    float b = aAndB.second;
    //gout = (gin-a)*255/(b-a)
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            float val = img(i, j);
            normalized(i, j) = static_cast<uchar>((val - a) * 255.0f / (b - a));
        }
    }
    return normalized;
}
Mat_<uchar> applyAbsoluteNormalization(Mat_<float> img, Mat_<float> kernel) {
    Mat_<uchar> normalized(img.size());
    pair<float, float> aAndB = computeAandB(kernel);
    float a = aAndB.first;
    float b = aAndB.second;
    //gout = abs(gin*255/max(abs(a),b))
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            float val = img(i, j);
            normalized(i,j)=static_cast<uchar>(abs(val*255/max(abs(a),b)));
        }
    }
    return normalized;
}

Mat_<float> createMeanKernel3x3() {
    Mat_<float> k = (Mat_<float>(3, 3) <<
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    );

    k /= 9.0f;
    return k;
}

Mat_<float> createGaussianKernel3x3() {
    Mat_<float> k = (Mat_<float>(3, 3) <<
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    );

    k /= 16.0f;
    return k;
}

Mat_<float> createLaplace4Kernel3x3() {
    return (Mat_<float>(3, 3) <<
         0, -1,  0,
        -1,  4, -1,
         0, -1,  0
    );
}

Mat_<float> createLaplace8Kernel3x3() {
    return (Mat_<float>(3, 3) <<
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    );
}

Mat_<float> createHighPass4Kernel3x3() {
    return (Mat_<float>(3, 3) <<
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    );
}

Mat_<float> createHighPass8Kernel3x3() {
    return (Mat_<float>(3, 3) <<
        -1, -1, -1,
        -1,  9, -1,
        -1, -1, -1
    );
}

void applyConvol(Mat_<uchar> img, Mat_<float> kernel) {
    Mat_<float> convol_img=convolution(img, kernel);
    Mat_<uchar> normalized_img_raw=applyRawNormalization(convol_img,kernel);
    Mat_<uchar> normalized_img_abs=applyAbsoluteNormalization(convol_img, kernel);
    imshow("Original image", img);
    imshow("Raw image", normalized_img_raw);
    imshow("absolute image", normalized_img_abs);
    waitKey(0);
}

void lab9() {
    int op;
    do{
        printf("Menu:\n");
        printf(" 1 - Convolution \n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d",&op);
        switch (op)
        {
            case 1: {
                Mat_<uchar> img = imread("PI-L9/cameraman.bmp", IMREAD_GRAYSCALE);
                vector<Mat_<float>> kernels = {createMeanKernel3x3(), createGaussianKernel3x3(), createLaplace8Kernel3x3(), createLaplace4Kernel3x3(), createHighPass4Kernel3x3(), createHighPass8Kernel3x3()};
                for (Mat_<float> kernel : kernels) {
                    applyConvol(img,kernel);
                }

                break;
            }
        }
    }
    while (op!=0);
}

Mat_<float> ideal_low_pass_filter(Mat_<uchar> img, float r) {
    Mat_ <float> filter(img.size(), 0.0f);
    r=pow(r,2);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float a = ((float)img.rows/2 - i)*((float)img.rows/2 - i);
            float b = ((float)img.cols/2 - j)*((float)img.cols/2 - j);
            float left = a+b;
            if (left<=r) {
                filter(i,j)=1;
            }
            else {
                filter(i,j)=0;
            }
        }
    }
    return filter;
}
Mat_<float> ideal_high_pass_filter(Mat_<uchar> img, float r) {
    Mat_ <float> filter(img.size(), 0.0f);
    r=pow(r,2);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int a = ((float)img.rows/2 - i)*((float)img.rows/2 - i);
            int b = ((float)img.cols/2 - j)*((float)img.cols/2 - j);
            float left = a+b;
            if (left>r) {
                filter(i,j)=1;
            }
            else {
                filter(i,j)=0;
            }
        }
    }
    return filter;
}

Mat_<float> gaussian_low_pass_filter(Mat_<uchar> img, float a) {
    Mat_<float> dest(img.size(), 0.0f);
    // float d = standard_deviation(img);
    // float a = 1/d;
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            float x = ((float)img.rows/2 - i)*((float)img.rows/2 - i);
            float y = ((float)img.cols/2 - j)*((float)img.cols/2 - j);
            float exponent= (x+y)/(a*a)*(-1);
            dest(i,j)=exp(exponent);
        }
    }
    return dest;
}

Mat_<float> gaussian_high_pass_filter(Mat_<uchar> img, float a) {
    Mat_<float> dest(img.size(), 0.0f);
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            float x = ((float)img.rows/2 - i)*((float)img.rows/2 - i);
            float y = ((float)img.cols/2 - j)*((float)img.cols/2 - j);
            float exponent = (x+y)/(a*a)*(-1);
            dest(i,j)=1-exp(exponent);
        }
    }
    return dest;
}

void centering_transform(Mat img){
    //expects floating point image
    for (int i = 0; i < img.rows; i++){
        for (int j = 0; j < img.cols; j++){
            img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
        }
    }
}

Mat_<float> generic_frequency_domain_filter(Mat_<uchar> src, Mat_<float> filter, bool normalize_result) {
    //convert input image to float image
    Mat srcf;
    src.convertTo(srcf, CV_32FC1);
    //centering transformation
    centering_transform(srcf);
    //perform forward transform with complex image output
    Mat fourier;
    dft(srcf, fourier, DFT_COMPLEX_OUTPUT);
    //split into real and imaginary channels
    Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
    split(fourier, channels); // channels[0] = Re(DFT(I)), channels[1] = Im(DFT(I))
    //calculate magnitude and phase in floating point images mag and phi
    Mat mag, phi;
    magnitude(channels[0], channels[1], mag);
    phase(channels[0], channels[1], phi);
    //display the phase and magnitude images here
    // ......
    Mat mag2=mag.clone();
    mag2 += Scalar::all(1);//to avoid log(0)
    log(mag2, mag2);
    normalize(mag2, mag2, 0, 255, NORM_MINMAX, CV_8UC1);
    imshow("Magnitude", mag2);
    Mat phi2=phi.clone();
    normalize(phi2, phi2, 0, 255, NORM_MINMAX, CV_8UC1);
    imshow("Phase", phi2);
    //insert filtering operations on Fourier coefficients here
    // ......
    Mat mag3=mag.mul(filter);
    //store in real part in channels[0] and imaginary part in channels[1]
    // ......
    for (int i = 0; i < mag.rows; i++) {
        for (int j = 0; j < mag.cols; j++) {
            channels[0].at<float>(i, j) = mag3.at<float>(i, j) * cos(phi.at<float>(i, j));
            channels[1].at<float>(i, j) = mag3.at<float>(i, j) * sin(phi.at<float>(i, j));
        }
    }
    //perform inverse transform and put results in dstf
    Mat dst, dstf;
    merge(channels, 2, fourier);
    dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    //inverse centering transformation
    centering_transform(dstf);
    //normalize the result and put in the destination image
    //normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    //Note: normalizing distorts the resut while enhancing the image display in the range [0,255].
    //For exact results (see Practical work 3) the normalization should be replaced with convertion:
    if (normalize_result) {
        normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    }
    else {
        dstf.convertTo(dst, CV_8UC1);
    }
    return dst;
}

Mat_<uchar> crop_to_same_ratio(Mat_<uchar> img, Size target_size) {
    float target_ratio = (float)target_size.width / target_size.height;
    float img_ratio = (float)img.cols / img.rows;
    Rect roi; //region of interest
    if (img_ratio > target_ratio) {
        int new_width = (int)(img.rows * target_ratio);
        int x = (img.cols - new_width) / 2;
        roi = Rect(x, 0, new_width, img.rows);
    }
    else {
        int new_height = (int)(img.cols / target_ratio);
        int y = (img.rows - new_height) / 2;
        roi = Rect(0, y, img.cols, new_height);
    }
    Mat cropped = img(roi).clone();
    Mat resized;
    resize(cropped, resized, target_size);
    return resized;
}

void hybrid_image_utcn_eth() {
    Mat_<uchar> utcn_initial = imread("Img/utcn.jpg", IMREAD_GRAYSCALE);
    Mat_<uchar> eth_initial = imread("Img/ETH.jpg", IMREAD_GRAYSCALE);
    Size target_size(eth_initial.cols, eth_initial.rows);
    Mat_<uchar> utcn = crop_to_same_ratio(utcn_initial, target_size);
    Mat_<uchar> eth = eth_initial.clone();

    Mat_<float> low_filter = gaussian_low_pass_filter(utcn, 25.0f);
    Mat_<float> high_filter = gaussian_high_pass_filter(eth, 12.0f);

    Mat_<float> utcn_low = generic_frequency_domain_filter(utcn, low_filter,1);
    Mat_<float> eth_high = generic_frequency_domain_filter(eth, high_filter, 1);

    Mat_<float> hybrid_float = 0.53f*utcn_low + 0.47f * eth_high;

    Mat hybrid;
    hybrid_float.convertTo(hybrid, CV_8UC1);

    imshow("UTCN resized", utcn);
    imshow("ETH resized", eth);
    imshow("Hybrid image", hybrid);

    waitKey(0);
}

void lab10() {
    int op;
    do{
        printf("Menu:\n");
        printf(" 1 - Same picture \n");
        printf(" 2 - Ideal Low Pass Filter \n");
        printf(" 3 - Ideal High Pass Filter \n");
        printf(" 4 - Gaussian Low Pass Filter \n");
        printf(" 5 - Gaussian High Pass Filter \n");
        printf(" 6 - Current vs DREAM \n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d",&op);
        switch (op)
        {
            case 1: {
                Mat_<uchar> img = imread("PI-L9/cameraman.bmp", IMREAD_GRAYSCALE);
                Mat filter(img.size(), CV_32FC1, Scalar(1));
                Mat dst = generic_frequency_domain_filter(img, filter,0);
                imshow("Filtered Image", dst);
                waitKey(0);
                break;
            }
            case 2: {
                Mat_<uchar> img = imread("PI-L9/cameraman.bmp", IMREAD_GRAYSCALE);
                Mat filter = ideal_low_pass_filter(img, 30.0f);
                Mat dst = generic_frequency_domain_filter(img, filter,0);
                imshow("initial image", img);
                imshow("Ideal Low Pass Filtered Image", dst);
                waitKey(0);
                break;
            }
            case 3: {
                Mat_<uchar> img = imread("PI-L9/cameraman.bmp", IMREAD_GRAYSCALE);
                Mat filter = ideal_high_pass_filter(img, 30.0f);
                Mat dst = generic_frequency_domain_filter(img, filter,1);
                imshow("initial image", img);
                imshow("Ideal High Pass Filtered Image", dst);
                waitKey(0);
                break;
            }
            case 4: {
                Mat_<uchar> img = imread("PI-L9/cameraman.bmp", IMREAD_GRAYSCALE);
                Mat filter = gaussian_low_pass_filter(img, 40.0f);
                Mat dst = generic_frequency_domain_filter(img, filter,1);
                imshow("initial image", img);
                imshow("Gaussian Low Pass Filtered Image", dst);
                waitKey(0);
                break;
            }
            case 5: {
                Mat_<uchar> img = imread("PI-L9/cameraman.bmp", IMREAD_GRAYSCALE);
                Mat filter = gaussian_high_pass_filter(img, 40.0f);
                Mat dst = generic_frequency_domain_filter(img, filter,1);
                imshow("Initial image", img);
                imshow("Gaussian High Pass Filtered Image", dst);
                waitKey(0);
                break;
            }
            case 6: {
                hybrid_image_utcn_eth();
                waitKey(0);
                break;
            }
        }
    }
    while (op!=0);
}

Mat_<float> createGaussianKernel2D(int w) {
    Mat_<float> kernel(w, w);
    float sigma = w / 6.0f;
    int center = w / 2;
    float sum = 0.0f;
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < w; j++) {
            float x = j - center;
            float y = i - center;
            kernel(i, j) = (1.0f / (2.0f * CV_PI * sigma * sigma)) * exp(-(x * x + y * y) / (2.0f * sigma * sigma));
            sum += kernel(i, j);
        }
    }
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < w; j++) {
            kernel(i, j) /= sum;
        }
    }
    return kernel;
}

Mat_<float> createGaussianKernel1D(int w) {
    Mat_<float> kernel(1, w);
    float sigma = w / 6.0f;
    int center = w / 2;
    float sum = 0.0f;
    for (int j = 0; j < w; j++) {
        float x = j - center;
        kernel(0, j) = (1.0f / (sqrt(2.0f * CV_PI) * sigma)) * exp(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel(0, j);
    }
    for (int j = 0; j < w; j++) {
        kernel(0, j) /= sum;
    }
    return kernel;
}

Mat_<uchar> applyGaussian2D(Mat_<float> kernel, Mat_<uchar> img, int w) {
    Mat_<uchar> result(img.rows, img.cols);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float sum = 0.0f;
            for (int u = 0; u < w; u++) {
                for (int v = 0; v < w; v++) {
                    int i2 = i + u - w / 2;
                    int j2 = j + v - w / 2;
                    if (isInside(img, i2, j2)) {
                        sum += img(i2, j2) * kernel(u, v);
                    }
                }
            }
            result(i, j) = saturate((int)round(sum));
        }
    }
    return result;
}

void Gaussian_2D(Mat_<uchar> img) {
    int w;
    cout << "Enter Gaussian kernel size w = 3, 5 or 7: ";
    cin >> w;
    Mat_<float> kernel = createGaussianKernel2D(w);
    double t = (double)getTickCount();
    Mat_<uchar> result = applyGaussian2D(kernel, img, w);
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "Gaussian 2D time = " << t * 1000 << " ms\n";
    imshow("Original image", img);
    imshow("Gaussian 2D filtered image", result);
    waitKey(0);
}

Mat_<float> createGaussianKernelGx(int w) {
    Mat_<float> kernel(1, w);
    float sigma = w / 6.0f;
    int center = w / 2;
    float sum = 0.0f;
    for (int x = 0; x < w; x++) {
        kernel(0, x) = (1.0f / (sqrt(2.0f * CV_PI) * sigma)) * exp(-((x - center) * (x - center)) / (2.0f * sigma * sigma));
        sum += kernel(0, x);
    }
    for (int x = 0; x < w; x++) {
        kernel(0, x) /= sum;
    }
    return kernel;
}

Mat_<float> createGaussianKernelGy(int w) {
    Mat_<float> kernel(w, 1);
    float sigma = w / 6.0f;
    int center = w / 2;
    float sum = 0.0f;
    for (int y = 0; y < w; y++) {
        kernel(y, 0) = (1.0f / (sqrt(2.0f * CV_PI) * sigma)) * exp(-((y - center) * (y - center)) / (2.0f * sigma * sigma));
        sum += kernel(y, 0);
    }
    for (int y = 0; y < w; y++) {
        kernel(y, 0) /= sum;
    }
    return kernel;
}

Mat_<uchar> applyGaussian1D(Mat_<uchar> img, int w, Mat_<float> Gx, Mat_<float> Gy) {
    Mat_<float> temp(img.rows, img.cols);
    Mat_<uchar> result(img.rows, img.cols);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float sum = 0.0f;
            for (int v = 0; v < w; v++) {
                int j2 = j + v - w / 2;
                if (isInside(img, i, j2)) {
                    sum += img(i, j2) * Gx(0, v);
                }
            }
            temp(i, j) = sum;
        }
    }
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            float sum = 0.0f;
            for (int u = 0; u < w; u++) {
                int i2 = i + u - w / 2;
                if (i2 >= 0 && i2 < img.rows) {
                    sum += temp(i2, j) * Gy(u, 0);
                }
            }
            result(i, j) = saturate((int)round(sum));
        }
    }
    return result;
}

void Gaussian_1D(Mat_<uchar> img) {
    int w;
    cout << "Enter Gaussian kernel size w = 3, 5 or 7: ";
    cin >> w;
    Mat_<float> Gx = createGaussianKernelGx(w);
    Mat_<float> Gy = createGaussianKernelGy(w);
    double t = (double)getTickCount();
    Mat_<uchar> result = applyGaussian1D(img, w, Gx, Gy);
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "Gaussian 1D separated time = " << t * 1000 << " ms\n";
    imshow("Original image", img);
    imshow("Gaussian 1D filtered image", result);
    waitKey(0);
}

Mat_<uchar> applyMedianFilter(Mat_<uchar> img, int w) {
    Mat_<uchar> result(img.rows, img.cols);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            vector<uchar> values;
            for (int u = 0; u < w; u++) {
                for (int v = 0; v < w; v++) {
                    int i2 = i + u - w / 2;
                    int j2 = j + v - w / 2;
                    if (isInside(img, i2, j2)) {
                        values.push_back(img(i2, j2));
                    }
                }
            }
            sort(values.begin(), values.end());
            result(i, j) = values[values.size() / 2];
        }
    }
    return result;
}

void Median_Filter(Mat_<uchar> img) {
    int w;
    cout << "Enter median filter size w = 3, 5 or 7: ";
    cin >> w;
    double t = (double)getTickCount();
    Mat_<uchar> result = applyMedianFilter(img, w);
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << "Median filter time = " << t * 1000 << " ms\n";
    imshow("Original image", img);
    imshow("Median filtered image", result);
    waitKey(0);
}

void lab11() {
    int op;
    do{
        printf("Menu:\n");
        printf(" 1 - Gaussian 2D \n");
        printf(" 2 - Gaussian 1D \n");
        printf(" 3 - Median filter \n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d",&op);
        switch (op)
        {
            case 1: {
                Mat_ <uchar> img2 = imread("PI-L10/portrait_Gauss2.bmp", IMREAD_GRAYSCALE);
                Gaussian_2D(img2);
                Mat_ <uchar> img = imread("PI-L10/balloons_Gauss.bmp", IMREAD_GRAYSCALE);
                Gaussian_2D(img);
                break;
            }
            case 2: {
                Mat_ <uchar> img2 = imread("PI-L10/portrait_Gauss2.bmp", IMREAD_GRAYSCALE);
                Gaussian_1D(img2);
                Mat_ <uchar> img = imread("PI-L10/balloons_Gauss.bmp", IMREAD_GRAYSCALE);
                Gaussian_1D(img);
                break;
            }
            case 3: {
                Mat_ <uchar> img2 = imread("PI-L10/portrait_Gauss2.bmp", IMREAD_GRAYSCALE);
                Median_Filter(img2);
                Mat_ <uchar> img = imread("PI-L10/balloons_Gauss.bmp", IMREAD_GRAYSCALE);
                Median_Filter(img);
                break;
            }
        }
    }
    while (op!=0);
}
const float STRONG_EDGE = 255.0f;
const float WEAK_EDGE = 128.0f;
const float NON_EDGE = 0.0f;
const float SOBEL_SCALE = 4.0f * sqrt(2.0f);

bool isInsideFloat(const Mat_<float>& img, int i, int j)
{
    return i >= 0 && i < img.rows && j >= 0 && j < img.cols;
}

Mat_<float> applyGaussianForCanny(Mat_<uchar> img)
{
    int w = 3; // sigma = w / 6 = 0.5

    Mat_<float> Gx = createGaussianKernelGx(w);
    Mat_<float> Gy = createGaussianKernelGy(w);

    Mat_<uchar> gaussianUchar = applyGaussian1D(img, w, Gx, Gy);

    Mat_<float> gaussian(gaussianUchar.rows, gaussianUchar.cols);
    gaussian.setTo(0);

    for (int i = 0; i < gaussianUchar.rows; i++)
    {
        for (int j = 0; j < gaussianUchar.cols; j++)
        {
            gaussian(i, j) = (float)gaussianUchar(i, j);
        }
    }

    return gaussian;
}

Mat_<float> sobelX(Mat_<float> img)
{
    Mat_<float> result(img.rows, img.cols);
    result.setTo(0);

    int kernel[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    for (int i = 1; i < img.rows - 1; i++)
    {
        for (int j = 1; j < img.cols - 1; j++)
        {
            float sum = 0.0f;

            for (int u = 0; u < 3; u++)
            {
                for (int v = 0; v < 3; v++)
                {
                    int x = i + u - 1;
                    int y = j + v - 1;

                    sum += img(x, y) * kernel[u][v];
                }
            }

            result(i, j) = sum;
        }
    }

    return result;
}

Mat_<float> sobelY(Mat_<float> img)
{
    Mat_<float> result(img.rows, img.cols);
    result.setTo(0);

    int kernel[3][3] = {
        { 1,  2,  1},
        { 0,  0,  0},
        {-1, -2, -1}
    };

    for (int i = 1; i < img.rows - 1; i++)
    {
        for (int j = 1; j < img.cols - 1; j++)
        {
            float sum = 0.0f;

            for (int u = 0; u < 3; u++)
            {
                for (int v = 0; v < 3; v++)
                {
                    int x = i + u - 1;
                    int y = j + v - 1;

                    sum += img(x, y) * kernel[u][v];
                }
            }

            result(i, j) = sum;
        }
    }

    return result;
}

Mat_<float> gradientMagnitude(Mat_<float> dx, Mat_<float> dy)
{
    Mat_<float> result(dx.rows, dx.cols);
    result.setTo(0);

    for (int i = 0; i < dx.rows; i++)
    {
        for (int j = 0; j < dx.cols; j++)
        {
            result(i, j) = sqrt(dx(i, j) * dx(i, j) + dy(i, j) * dy(i, j));
        }
    }

    return result;
}

Mat_<float> gradientDirection(Mat_<float> dx, Mat_<float> dy)
{
    Mat_<float> result(dx.rows, dx.cols);
    result.setTo(0);

    for (int i = 0; i < dx.rows; i++)
    {
        for (int j = 0; j < dx.cols; j++)
        {
            result(i, j) = atan2(dy(i, j), dx(i, j));
        }
    }

    return result;
}

Mat_<uchar> computeQ(Mat_<float> direction)
{
    Mat_<uchar> q(direction.rows, direction.cols);
    q.setTo(0);

    int mapToFourRegions[8] = {
        2, // 0 degrees
        1, // 45 degrees
        0, // 90 degrees
        3, // 135 degrees
        2, // 180 degrees
        1, // 225 degrees
        0, // 270 degrees
        3  // 315 degrees
    };

    for (int i = 0; i < direction.rows; i++)
    {
        for (int j = 0; j < direction.cols; j++)
        {
            float phi = direction(i, j);

            if (phi < 0)
            {
                phi += 2.0f * CV_PI;
            }

            int sector = (int)(phi * 8.0f / (2.0f * CV_PI) + 0.5f);
            sector = sector % 8;

            q(i, j) = mapToFourRegions[sector];
        }
    }

    return q;
}

Mat_<float> nonMaximaSuppression(Mat_<float> magnitude, Mat_<uchar> q)
{
    Mat_<float> thinnedMag(magnitude.rows, magnitude.cols);
    thinnedMag.setTo(0);

    for (int i = 1; i < magnitude.rows - 1; i++)
    {
        for (int j = 1; j < magnitude.cols - 1; j++)
        {
            float current = magnitude(i, j);
            float neighbor1 = 0.0f;
            float neighbor2 = 0.0f;

            if (q(i, j) == 0)
            {
                // vertical: compare up and down
                neighbor1 = magnitude(i - 1, j);
                neighbor2 = magnitude(i + 1, j);
            }
            else if (q(i, j) == 1)
            {
                // diagonal 45 / 225: compare NE and SW
                neighbor1 = magnitude(i - 1, j + 1);
                neighbor2 = magnitude(i + 1, j - 1);
            }
            else if (q(i, j) == 2)
            {
                // horizontal: compare left and right
                neighbor1 = magnitude(i, j - 1);
                neighbor2 = magnitude(i, j + 1);
            }
            else if (q(i, j) == 3)
            {
                // diagonal 135 / 315: compare NW and SE
                neighbor1 = magnitude(i - 1, j - 1);
                neighbor2 = magnitude(i + 1, j + 1);
            }

            if (current > neighbor1 && current > neighbor2)
            {
                thinnedMag(i, j) = current;
            }
            else
            {
                thinnedMag(i, j) = 0.0f;
            }
        }
    }

    return thinnedMag;
}

int scaledSobelValue(float value)
{
    int scaled = (int)(value / SOBEL_SCALE);

    if (scaled < 0)
    {
        scaled = 0;
    }

    if (scaled > 255)
    {
        scaled = 255;
    }

    return scaled;
}

Mat_<float> scaleSobelMagnitudeFloat(Mat_<float> mag)
{
    Mat_<float> result(mag.rows, mag.cols);
    result.setTo(0);

    for (int i = 0; i < mag.rows; i++)
    {
        for (int j = 0; j < mag.cols; j++)
        {
            result(i, j) = (float)scaledSobelValue(mag(i, j));
        }
    }

    return result;
}

float adaptiveThresholding(Mat_<float> thinnedMag, float p)
{
    vector<int> hist(256, 0);

    for (int i = 0; i < thinnedMag.rows; i++)
    {
        for (int j = 0; j < thinnedMag.cols; j++)
        {
            int value = scaledSobelValue(thinnedMag(i, j));
            hist[value]++;
        }
    }

    int noNonZeroPixels = thinnedMag.rows * thinnedMag.cols - hist[0];

    if (noNonZeroPixels == 0)
    {
        return 0.0f;
    }

    int noNonEdge = (int)((1.0f - p) * noNonZeroPixels);

    int sum = 0;

    for (int i = 1; i < 256; i++)
    {
        sum += hist[i];

        if (sum > noNonEdge)
        {
            return (float)i;
        }
    }

    return 255.0f;
}

Mat_<float> adaptiveThresholdResult(Mat_<float> thinnedMag, float threshold)
{
    Mat_<float> result(thinnedMag.rows, thinnedMag.cols);
    result.setTo(0);

    for (int i = 0; i < thinnedMag.rows; i++)
    {
        for (int j = 0; j < thinnedMag.cols; j++)
        {
            int value = scaledSobelValue(thinnedMag(i, j));

            if (value >= threshold)
            {
                result(i, j) = STRONG_EDGE;
            }
            else
            {
                result(i, j) = NON_EDGE;
            }
        }
    }

    return result;
}

Mat_<float> hysteresisThresholding(Mat_<float> thinnedMag, float thresholdHigh, float k)
{
    Mat_<float> result(thinnedMag.rows, thinnedMag.cols);
    result.setTo(0);

    float thresholdLow = k * thresholdHigh;

    for (int i = 0; i < thinnedMag.rows; i++)
    {
        for (int j = 0; j < thinnedMag.cols; j++)
        {
            int value = scaledSobelValue(thinnedMag(i, j));

            if (value >= thresholdHigh)
            {
                result(i, j) = STRONG_EDGE;
            }
            else if (value >= thresholdLow)
            {
                result(i, j) = WEAK_EDGE;
            }
            else
            {
                result(i, j) = NON_EDGE;
            }
        }
    }

    return result;
}

Mat_<float> edgeExtensionHysteresis(Mat_<float> edges)
{
    Mat_<float> result = edges.clone();
    queue<Point> Q;
    int di[8] = {-1, -1, -1,  0, 0,  1, 1, 1};
    int dj[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            if (result(i, j) == STRONG_EDGE)
            {
                Q.push(Point(j, i));
                while (!Q.empty())
                {
                    Point current = Q.front();
                    Q.pop();
                    for (int d = 0; d < 8; d++)
                    {
                        int ni = current.y + di[d];
                        int nj = current.x + dj[d];
                        if (isInsideFloat(result, ni, nj) && result(ni, nj) == WEAK_EDGE)
                        {
                            result(ni, nj) = STRONG_EDGE;
                            Q.push(Point(nj, ni));
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            if (result(i, j) == WEAK_EDGE)
            {
                result(i, j) = NON_EDGE;
            }
        }
    }
    return result;
}

void cannyGradientStep()
{
    Mat_<uchar> img = imread("PI-L11/cameraman.bmp", IMREAD_GRAYSCALE);

    Mat_<float> gaussian = applyGaussianForCanny(img);

    Mat_<float> dx = sobelX(gaussian);
    Mat_<float> dy = sobelY(gaussian);

    Mat_<float> magnitude = gradientMagnitude(dx, dy);
    Mat_<float> direction = gradientDirection(dx, dy);

    Mat_<uchar> q = computeQ(direction);

    Mat_<float> thinnedMag = nonMaximaSuppression(magnitude, q);

    float p = 0.1f;
    float k = 0.4f;

    float thresholdHigh = adaptiveThresholding(thinnedMag, p);
    float thresholdLow = k * thresholdHigh;

    cout << "Threshold high: " << thresholdHigh << endl;
    cout << "Threshold low: " << thresholdLow << endl;

    Mat_<float> adaptiveImage = adaptiveThresholdResult(thinnedMag, thresholdHigh);
    Mat_<float> hysteresisImage = hysteresisThresholding(thinnedMag, thresholdHigh, k);
    Mat_<float> finalEdges = edgeExtensionHysteresis(hysteresisImage);

    imshow("Initial image", img);
    imshow("Gaussian image", gaussian / 255.0f);
    imshow("Normalized gradient magnitude", scaleSobelMagnitudeFloat(magnitude) / 255.0f);
    imshow("After non-maxima suppression", scaleSobelMagnitudeFloat(thinnedMag) / 255.0f);
    imshow("After adaptive thresholding", adaptiveImage / 255.0f);
    imshow("Strong and weak edges", hysteresisImage / 255.0f);
    imshow("Final Canny edges", finalEdges / 255.0f);

    waitKey(0);
}

void lab12()
{
    cannyGradientStep();
}

void negative_image(){
    Mat_<uchar> img = imread("Images/cameraman.bmp",
     IMREAD_GRAYSCALE);
    for(int i=0; i<img.rows; i++){
        for(int j=0; j<img.cols; j++){
            img.at<uchar>(i,j) = 255 - img.at<uchar>(i,j);
        }
    }
    imshow("negative image",img);
    waitKey(0);
}
void additive_factor(int k) {
    Mat_<uchar> img = imread("Images/cameraman.bmp",IMREAD_GRAYSCALE);
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            if (k>0) {
                img.at<uchar>(i,j)=min(255, img.at<uchar>(i,j)+k);
            }
            else {
                img.at<uchar>(i,j)=max(0,img.at<uchar>(i,j)+k);
            }
        }
    }

    imshow("additive factor", img);
    waitKey(0);
}
void multiplicative_factor(int k) {
    Mat_<uchar> img = imread("Images/cameraman.bmp", IMREAD_GRAYSCALE);
    for (int i=0; i<img.rows; i++) {
        for (int j=0; j<img.cols; j++) {
            if (k>0) {
                img.at<uchar>(i,j)=min(255, img.at<uchar>(i,j)*k);
            }
            else {
                img.at<uchar>(i,j)=max(0,img.at<uchar>(i,j)*k);
            }
        }
    }
    //save image
    imwrite("fname.bmp", img);

    imshow("multiplicative factor", img);
    waitKey(0);
}
void build_color_image() {
    Mat_<Vec3b> img(256,256);

    for (int i=0; i<img.rows/2; i++) {
        for (int j=0; j<img.cols/2; j++) {
            //white
            img(i,j) = {255, 255, 255};
        }
    }
    for (int i=0; i<img.rows/2; i++) {
        for (int j=img.cols/2; j<img.cols; j++) {
            //red
            img(i, j) = {0, 0, 255}; //blue, green, red
        }
    }
    for (int i=img.rows/2; i<img.rows; i++) {
        for (int j=0; j<img.cols/2; j++) {
            //green
            img(i,j)={0,255,0}; //blue, green, red
        }
    }
    for (int i=img.rows/2; i<img.rows; i++) {
        for (int j=img.cols/2; j<img.cols; j++) {
            //yellow
            img(i,j)={0, 255, 255};
        }
    }

    imshow("colored image", img);
    waitKey();
}
void inverse_float_matrix() {
    Mat_<float> floatMatrix = (Mat_<float>(3,3) <<
        12.5f, 45.2f, 78.9f,
        34.1f, 56.7f, 91.3f,
        27.4f, 63.8f, 15.6f
    );

    cout<<"Initial Matrix: \n";
    for (int i=0; i<floatMatrix.rows; i++) {
        for (int j=0; j<floatMatrix.cols; j++) {
            cout<<floatMatrix.at<float>(i,j)<<" ";
        }
        cout<<endl;
    }

    cout<<endl<<"Inverse Matrix: \n";
    Mat_<float> inverseMatrix = floatMatrix.inv(DECOMP_LU);
    for (int i=0; i<inverseMatrix.rows; i++) {
        for (int j=0; j<inverseMatrix.cols; j++) {
            cout<<inverseMatrix.at<float>(i,j)<<' ';
        }
        cout<<endl;
    }

    //cout << "A * invA =\n" << floatMatrix * inverseMatrix << endl;
}
void lab1_main() {
    int op;
    do{
        printf("Menu:\n");
        printf(" 1 - Testing Negative Image Function \n");
        printf(" 2 - Change the gray levels of an image by an additive factor \n");
        printf(" 3 - Change the gray levels of an image by a multiplicative factor \n");
        printf(" 4 - Color image of dimension 256x256 colored in white, red, green, yellow \n");
        printf(" 5 - Create a 3x3 floating matrix, determine its inverse and print it \n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d",&op);
        switch (op)
        {
            case 1:
                negative_image();
                break;
            case 2:
                additive_factor(100);
                break;
            case 3:
                multiplicative_factor(2);
                break;
            case 4:
                build_color_image();
                break;
            case 5:
                inverse_float_matrix();
                break;
        }
    }
    while (op!=0);
}

// PROJECT

void splitHSVChannels(Mat_<Vec3b> hsv, Mat_<uchar>& H, Mat_<uchar>& S, Mat_<uchar>& V)
{
    H = Mat_<uchar>(hsv.rows, hsv.cols);
    S = Mat_<uchar>(hsv.rows, hsv.cols);
    V = Mat_<uchar>(hsv.rows, hsv.cols);
    for (int i = 0; i < hsv.rows; i++) {
        for (int j = 0; j < hsv.cols; j++) {
            H(i, j) = hsv(i, j)[0];
            S(i, j) = hsv(i, j)[1];
            V(i, j) = hsv(i, j)[2];
        }
    }
}

struct ComponentInfo {
    int label = 0;
    int area = 0;
    Point2f center = Point2f(0, 0);
    Rect bbox;
    float aspect = 0.0f;
    float bboxDensity = 0.0f;
    int meanH = 0;
    int meanS = 0;
    int meanV = 0;
    int meanB = 0;
    int meanG = 0;
    int meanR = 0;
    string colorName = "UNKNOWN";
};

Mat_<uchar> squareStrel3()
{
    Mat_<uchar> strel(3, 3);
    strel.setTo(0);
    return strel;
}

Mat_<uchar> squareStrel5()
{
    Mat_<uchar> strel(5, 5);
    strel.setTo(0);
    return strel;
}

Mat_<uchar> openingOp(Mat_<uchar> src, Mat_<uchar> strel)
{
    Mat_<uchar> eroded = erotion(src, strel);
    Mat_<uchar> opened = dilation(eroded, strel);
    return opened;
}

Mat_<uchar> closingOp(Mat_<uchar> src, Mat_<uchar> strel)
{
    Mat_<uchar> dilated = dilation(src, strel);
    Mat_<uchar> closed = erotion(dilated, strel);
    return closed;
}

Mat_<Vec3b> resizeProjectImageIfNeeded(Mat_<Vec3b> img)
{
    int maxDim = max(img.rows, img.cols);
    if (maxDim <= 900) {
        return img;
    }
    float scale = 900.0f / maxDim;
    Mat_<Vec3b> resized;
    resize(img, resized, Size(), scale, scale, INTER_AREA);
    return resized;
}

bool isSkinPixel(int H, int S, int V, int R, int G, int B)
{
    if (V < 45) {
        return false;
    }
    float gRatio = 0.0f;
    float bRatio = 0.0f;
    if (R > 0) {
        gRatio = (float)G / (float)R;
        bRatio = (float)B / (float)R;
    }
    bool hueSkin = H >= 0 && H <= 35;
    bool rgbSkin = R > 70 && G > 45 && B > 25 && R >= G && G >= B - 5 && R > B + 15;
    bool ratioSkin = gRatio >= 0.48f && gRatio <= 0.95f && bRatio >= 0.25f && bRatio <= 0.85f;
    bool saturationSkin = S >= 15 && S <= 135;
    return hueSkin && rgbSkin && ratioSkin && saturationSkin;
}

bool isWhiteRubikPixel(int H, int S, int V, int R, int G, int B)
{
    int maxRGB = max(R, max(G, B));
    int minRGB = min(R, min(G, B));
    bool brightEnough = V >= 105 && maxRGB >= 115 && R >= 85 && G >= 85 && B >= 85;
    bool lowSaturation = S <= 85;
    bool balancedRGB = maxRGB - minRGB <= 75;
    bool notSkin = !isSkinPixel(H, S, V, R, G, B);
    return brightEnough && lowSaturation && balancedRGB && notSkin;
}

bool isYellowRubikPixel(int H, int S, int V, int R, int G, int B)
{
    float gRatio = 0.0f;
    float bRatio = 0.0f;
    if (R > 0) {
        gRatio = (float)G / (float)R;
        bRatio = (float)B / (float)R;
    }
    bool hueOK = H >= 28 && H <= 82;
    bool rgbOK = R >= 95 && G >= 90 && B <= 170 && R > B + 30 && G > B + 30;
    bool ratioOK = gRatio >= 0.70f && gRatio <= 1.20f && bRatio <= 0.70f;
    bool satOK = S >= 45 && V >= 70;
    return hueOK && rgbOK && ratioOK && satOK;
}

bool isOrangeRubikPixel(int H, int S, int V, int R, int G, int B)
{
    float gRatio = 0.0f;
    float bRatio = 0.0f;
    if (R > 0) {
        gRatio = (float)G / (float)R;
        bRatio = (float)B / (float)R;
    }
    bool hueOK = H >= 7 && H <= 45;
    bool rgbOK = R >= 90 && G >= 40 && R > G + 5 && R > B + 22 && G > B + 5;
    bool ratioOK = gRatio >= 0.43f && gRatio <= 0.93f && bRatio <= 0.78f;
    bool satOK = S >= 55 && V >= 45;
    return hueOK && rgbOK && ratioOK && satOK;
}

bool isRedRubikPixel(int H, int S, int V, int R, int G, int B)
{
    float gRatio = 0.0f;
    float bRatio = 0.0f;
    if (R > 0) {
        gRatio = (float)G / (float)R;
        bRatio = (float)B / (float)R;
    }
    bool hueOK = H <= 14 || H >= 240;
    bool rgbOK = R >= 85 && R > G + 25 && R > B + 25;
    bool ratioOK = gRatio <= 0.62f && bRatio <= 0.85f;
    bool satOK = S >= 55 && V >= 45;
    return hueOK && rgbOK && ratioOK && satOK;
}

bool isGreenRubikPixel(int H, int S, int V, int R, int G, int B)
{
    bool hueOK = H >= 65 && H <= 155;
    bool rgbOK = G >= 55 && G > R + 8 && G > B + 3;
    bool satOK = S >= 45 && V >= 45;
    return hueOK && rgbOK && satOK;
}

bool isBlueRubikPixel(int H, int S, int V, int R, int G, int B)
{
    bool hueOK = H >= 110 && H <= 235;
    bool rgbOK = B >= 50 && B > R + 8 && B >= G - 10;
    bool satOK = S >= 40 && V >= 35;
    return hueOK && rgbOK && satOK;
}

string classifyRubikColor(int H, int S, int V, int R, int G, int B)
{
    if (isWhiteRubikPixel(H, S, V, R, G, B)) {
        return "WHITE";
    }
    if (isYellowRubikPixel(H, S, V, R, G, B)) {
        return "YELLOW";
    }
    if (isOrangeRubikPixel(H, S, V, R, G, B)) {
        return "ORANGE";
    }
    if (isRedRubikPixel(H, S, V, R, G, B)) {
        return "RED";
    }
    if (isGreenRubikPixel(H, S, V, R, G, B)) {
        return "GREEN";
    }
    if (isBlueRubikPixel(H, S, V, R, G, B)) {
        return "BLUE";
    }
    return "UNKNOWN";
}

bool isColoredRubikPixel(int H, int S, int V, int R, int G, int B)
{
    if (isSkinPixel(H, S, V, R, G, B)) {
        return false;
    }
    if (isYellowRubikPixel(H, S, V, R, G, B)) {
        return true;
    }
    if (isOrangeRubikPixel(H, S, V, R, G, B)) {
        return true;
    }
    if (isRedRubikPixel(H, S, V, R, G, B)) {
        return true;
    }
    if (isGreenRubikPixel(H, S, V, R, G, B)) {
        return true;
    }
    if (isBlueRubikPixel(H, S, V, R, G, B)) {
        return true;
    }
    return false;
}

Mat_<uchar> buildColoredRubikMask(Mat_<Vec3b> img, Mat_<uchar>& H, Mat_<uchar>& S, Mat_<uchar>& V)
{
    Mat_<Vec3b> hsv = convertRGBtoHSV(img);
    splitHSVChannels(hsv, H, S, V);
    Mat_<uchar> mask(img.rows, img.cols);
    mask.setTo(255);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int B = img(i, j)[0];
            int G = img(i, j)[1];
            int R = img(i, j)[2];
            if (isColoredRubikPixel(H(i, j), S(i, j), V(i, j), R, G, B)) {
                mask(i, j) = 0;
            }
        }
    }
    return mask;
}

Rect expandRectSafe(Rect r, int margin, Size size)
{
    int x1 = max(0, r.x - margin);
    int y1 = max(0, r.y - margin);
    int x2 = min(size.width - 1, r.x + r.width + margin);
    int y2 = min(size.height - 1, r.y + r.height + margin);
    return Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
}

vector<ComponentInfo> extractComponentsFromLabels(Mat_<int> labels, Mat_<uchar> H, Mat_<uchar> S, Mat_<uchar> V, Mat_<Vec3b> img)
{
    int maxLabel = 0;
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            if (labels(i, j) > maxLabel) {
                maxLabel = labels(i, j);
            }
        }
    }
    vector<int> area(maxLabel + 1, 0);
    vector<int> minX(maxLabel + 1, labels.cols);
    vector<int> minY(maxLabel + 1, labels.rows);
    vector<int> maxX(maxLabel + 1, 0);
    vector<int> maxY(maxLabel + 1, 0);
    vector<long long> sumX(maxLabel + 1, 0);
    vector<long long> sumY(maxLabel + 1, 0);
    vector<long long> sumH(maxLabel + 1, 0);
    vector<long long> sumS(maxLabel + 1, 0);
    vector<long long> sumV(maxLabel + 1, 0);
    vector<long long> sumB(maxLabel + 1, 0);
    vector<long long> sumG(maxLabel + 1, 0);
    vector<long long> sumR(maxLabel + 1, 0);
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            int label = labels(i, j);
            if (label > 0) {
                area[label]++;
                sumX[label] += j;
                sumY[label] += i;
                sumH[label] += H(i, j);
                sumS[label] += S(i, j);
                sumV[label] += V(i, j);
                sumB[label] += img(i, j)[0];
                sumG[label] += img(i, j)[1];
                sumR[label] += img(i, j)[2];
                if (j < minX[label]) minX[label] = j;
                if (j > maxX[label]) maxX[label] = j;
                if (i < minY[label]) minY[label] = i;
                if (i > maxY[label]) maxY[label] = i;
            }
        }
    }
    vector<ComponentInfo> components;
    for (int label = 1; label <= maxLabel; label++) {
        if (area[label] == 0) {
            continue;
        }
        ComponentInfo c;
        c.label = label;
        c.area = area[label];
        c.center = Point2f(
            (float)sumX[label] / area[label],
            (float)sumY[label] / area[label]
        );
        int width = maxX[label] - minX[label] + 1;
        int height = maxY[label] - minY[label] + 1;
        c.bbox = Rect(minX[label], minY[label], width, height);
        c.aspect = (float)width / height;
        int bboxArea = width * height;
        if (bboxArea > 0) {
            c.bboxDensity = (float)c.area / bboxArea;
        }
        c.meanH = (int)(sumH[label] / area[label]);
        c.meanS = (int)(sumS[label] / area[label]);
        c.meanV = (int)(sumV[label] / area[label]);
        c.meanB = (int)(sumB[label] / area[label]);
        c.meanG = (int)(sumG[label] / area[label]);
        c.meanR = (int)(sumR[label] / area[label]);
        c.colorName = classifyRubikColor(c.meanH, c.meanS, c.meanV, c.meanR, c.meanG, c.meanB);
        components.push_back(c);
    }
    return components;
}

bool isPossibleCubeFace(ComponentInfo c, Size imgSize)
{
    int imageArea = imgSize.width * imgSize.height;
    int minArea = (int)(0.008f * imageArea);
    int maxArea = (int)(0.40f * imageArea);

    if (c.area < minArea) return false;
    if (c.area > maxArea) return false;
    if (c.bbox.width < imgSize.width * 0.12f) return false;
    if (c.bbox.height < imgSize.height * 0.12f) return false;
    if (c.aspect < 0.50f || c.aspect > 2.40f) return false;
    if (c.bboxDensity < 0.16f) return false;
    if (c.colorName == "WHITE" || c.colorName == "UNKNOWN") return false;
    return true;
}

ComponentInfo findBestComponentFace(vector<ComponentInfo> components, Size imgSize)
{
    ComponentInfo best;
    float bestScore = -1.0f;
    for (ComponentInfo c : components) {
        if (!isPossibleCubeFace(c, imgSize)) {
            continue;
        }
        float areaScore = (float)c.area / (imgSize.width * imgSize.height);
        float densityScore = c.bboxDensity;
        float aspectPenalty = fabs(1.25f - c.aspect);
        float score = 20.0f * areaScore + 2.0f * densityScore - 0.5f * aspectPenalty;
        if (score > bestScore) {
            bestScore = score;
            best = c;
        }
    }
    return best;
}

string dominantColorInsideBox(Mat_<Vec3b> img, Mat_<uchar> H, Mat_<uchar> S, Mat_<uchar> V, Rect box)
{
    box = box & Rect(0, 0, img.cols, img.rows);
    vector<int> votes(6, 0);
    int marginX = max(2, box.width / 8);
    int marginY = max(2, box.height / 8);
    Rect inside(
        box.x + marginX,
        box.y + marginY,
        max(1, box.width - 2 * marginX),
        max(1, box.height - 2 * marginY)
    );
    for (int i = inside.y; i < inside.y + inside.height; i++) {
        for (int j = inside.x; j < inside.x + inside.width; j++) {
            int B = img(i, j)[0];
            int G = img(i, j)[1];
            int R = img(i, j)[2];
            string color = classifyRubikColor(H(i, j), S(i, j), V(i, j), R, G, B);
            if (color == "WHITE") votes[0]++;
            if (color == "YELLOW") votes[1]++;
            if (color == "RED") votes[2]++;
            if (color == "ORANGE") votes[3]++;
            if (color == "GREEN") votes[4]++;
            if (color == "BLUE") votes[5]++;
        }
    }
    int bestIndex = -1;
    int bestVotes = 0;
    for (int i = 0; i < 6; i++) {
        if (votes[i] > bestVotes) {
            bestVotes = votes[i];
            bestIndex = i;
        }
    }
    if (bestIndex == 0) return "WHITE";
    if (bestIndex == 1) return "YELLOW";
    if (bestIndex == 2) return "RED";
    if (bestIndex == 3) return "ORANGE";
    if (bestIndex == 4) return "GREEN";
    if (bestIndex == 5) return "BLUE";
    return "UNKNOWN";
}

float averageValueInRect(Mat_<uchar> img, Rect r)
{
    r = r & Rect(0, 0, img.cols, img.rows);
    if (r.width <= 0 || r.height <= 0) {
        return 0.0f;
    }
    long long sum = 0;
    int count = 0;
    for (int i = r.y; i < r.y + r.height; i++) {
        for (int j = r.x; j < r.x + r.width; j++) {
            sum += img(i, j);
            count++;
        }
    }
    if (count == 0) {
        return 0.0f;
    }
    return (float)sum / count;
}

float whiteDensityInRectForCanny(Mat_<Vec3b> img, Mat_<uchar> H, Mat_<uchar> S, Mat_<uchar> V, Rect r)
{
    r = r & Rect(0, 0, img.cols, img.rows);
    if (r.width <= 0 || r.height <= 0) {
        return 0.0f;
    }
    int whitePixels = 0;
    int totalPixels = 0;
    for (int i = r.y; i < r.y + r.height; i++) {
        for (int j = r.x; j < r.x + r.width; j++) {
            int B = img(i, j)[0];
            int G = img(i, j)[1];
            int R = img(i, j)[2];
            if (isWhiteRubikPixel(H(i, j), S(i, j), V(i, j), R, G, B)) {
                whitePixels++;
            }
            totalPixels++;
        }
    }
    if (totalPixels == 0) {
        return 0.0f;
    }
    return (float)whitePixels / totalPixels;
}

float cannyGridScore(Mat_<uchar> edges, Rect box)
{
    box = box & Rect(0, 0, edges.cols, edges.rows);
    if (box.width < 60 || box.height < 60) {
        return 0.0f;
    }
    int band = max(2, min(box.width, box.height) / 35);
    int x1 = box.x + box.width / 3;
    int x2 = box.x + 2 * box.width / 3;
    int y1 = box.y + box.height / 3;
    int y2 = box.y + 2 * box.height / 3;
    int edgeCount = 0;
    int totalCount = 0;
    for (int i = box.y; i < box.y + box.height; i++) {
        for (int d = -band; d <= band; d++) {
            int j1 = x1 + d;
            int j2 = x2 + d;
            if (isInside(edges, i, j1)) {
                if (edges(i, j1) != 0) edgeCount++;
                totalCount++;
            }
            if (isInside(edges, i, j2)) {
                if (edges(i, j2) != 0) edgeCount++;
                totalCount++;
            }
        }
    }
    for (int j = box.x; j < box.x + box.width; j++) {
        for (int d = -band; d <= band; d++) {
            int i1 = y1 + d;
            int i2 = y2 + d;
            if (isInside(edges, i1, j)) {
                if (edges(i1, j) != 0) edgeCount++;
                totalCount++;
            }
            if (isInside(edges, i2, j)) {
                if (edges(i2, j) != 0) edgeCount++;
                totalCount++;
            }
        }
    }
    if (totalCount == 0) {
        return 0.0f;
    }
    return (float)edgeCount / totalCount;
}

float outerEdgeScore(Mat_<uchar> edges, Rect box)
{
    box = box & Rect(0, 0, edges.cols, edges.rows);
    if (box.width < 60 || box.height < 60) {
        return 0.0f;
    }
    int band = max(2, min(box.width, box.height) / 35);
    int edgeCount = 0;
    int totalCount = 0;
    for (int j = box.x; j < box.x + box.width; j++) {
        for (int d = -band; d <= band; d++) {
            int top = box.y + d;
            int bottom = box.y + box.height - 1 + d;
            if (isInside(edges, top, j)) {
                if (edges(top, j) != 0) edgeCount++;
                totalCount++;
            }
            if (isInside(edges, bottom, j)) {
                if (edges(bottom, j) != 0) edgeCount++;
                totalCount++;
            }
        }
    }
    for (int i = box.y; i < box.y + box.height; i++) {
        for (int d = -band; d <= band; d++) {
            int left = box.x + d;
            int right = box.x + box.width - 1 + d;
            if (isInside(edges, i, left)) {
                if (edges(i, left) != 0) edgeCount++;
                totalCount++;
            }
            if (isInside(edges, i, right)) {
                if (edges(i, right) != 0) edgeCount++;
                totalCount++;
            }
        }
    }
    if (totalCount == 0) {
        return 0.0f;
    }
    return (float)edgeCount / totalCount;
}

ComponentInfo findWhiteFaceWithCanny(Mat_<Vec3b> img, Mat_<uchar> H, Mat_<uchar> S, Mat_<uchar> V, bool showCanny = true)
{
    ComponentInfo bestFace;
    Mat_<uchar> blurredV;
    GaussianBlur(V, blurredV, Size(5, 5), 1.0);
    Mat_<uchar> edges;
    Canny(blurredV, edges, 35, 110);
    if (showCanny) {
        imshow("White Canny on V channel", edges);
    }
    int minDim = min(img.rows, img.cols);
    int minSide = max(120, minDim / 5);
    int maxSide = min((int)(0.55f * minDim), minDim - 10);
    float bestScore = -1.0f;
    Rect bestBox;
    int sideStep = 12;
    int step = 8;
    int startX = img.cols * 10 / 100;
    for (int side = minSide; side <= maxSide; side += sideStep) {
        for (int y = 5; y + side < img.rows - 5; y += step) {
            for (int x = startX; x + side < img.cols - 5; x += step) {
                Rect box(x, y, side, side);
                float avgS = averageValueInRect(S, box);
                float avgV = averageValueInRect(V, box);
                float whiteDensity = whiteDensityInRectForCanny(img, H, S, V, box);
                if (avgS > 105.0f) {
                    continue;
                }
                if (avgV < 90.0f) {
                    continue;
                }
                if (whiteDensity < 0.15f) {
                    continue;
                }

                float gridScore = cannyGridScore(edges, box);
                float borderScore = outerEdgeScore(edges, box);
                if (gridScore < 0.012f && borderScore < 0.012f) {
                    continue;
                }
                Point2f center(
                    box.x + box.width / 2.0f,
                    box.y + box.height / 2.0f
                );
                float rightBonus = center.x / img.cols;
                float centerY = center.y / img.rows;
                float yBonus = 1.0f - fabs(centerY - 0.45f);
                if (yBonus < 0.0f) {
                    yBonus = 0.0f;
                }
                float lowSScore = max(0.0f, (105.0f - avgS) / 105.0f);
                float sizeScore = (float)side / minDim;
                float score =
                    whiteDensity * 5000.0f +
                    gridScore * 45000.0f +
                    borderScore * 18000.0f +
                    lowSScore * 3000.0f +
                    sizeScore * 2500.0f +
                    rightBonus * 1500.0f +
                    yBonus * 1000.0f;
                if (score > bestScore) {
                    bestScore = score;
                    bestBox = box;
                }
            }
        }
    }
    if (bestScore < 0.0f) {
        return bestFace;
    }
    bestBox = expandRectSafe(bestBox, 4, img.size());
    bestFace.label = -10;
    bestFace.bbox = bestBox;
    bestFace.area = bestBox.width * bestBox.height;
    bestFace.center = Point2f(
        bestBox.x + bestBox.width / 2.0f,
        bestBox.y + bestBox.height / 2.0f
    );
    bestFace.aspect = (float)bestBox.width / bestBox.height;
    bestFace.bboxDensity = 1.0f;
    bestFace.colorName = "WHITE";
    return bestFace;
}

struct StickerInfo {
    Rect box;
    RotatedRect rotatedBox;
    Point2f center;
    int row = -1;
    int col = -1;
    string color = "UNKNOWN";
    float confidence = 0.0f;
};

struct RubikFace {
    vector<Point2f> corners;
    Rect bbox;
    bool valid = false;
};

Vec3b colorNameToBGR(const string& colorName)
{
    if (colorName == "WHITE") {
        return Vec3b(255, 255, 255);
    }
    if (colorName == "YELLOW") {
        return Vec3b(0, 255, 255);
    }
    if (colorName == "RED") {
        return Vec3b(0, 0, 255);
    }
    if (colorName == "ORANGE") {
        return Vec3b(0, 165, 255);
    }
    if (colorName == "GREEN") {
        return Vec3b(0, 128, 0);
    }
    if (colorName == "BLUE") {
        return Vec3b(255, 0, 0);
    }
    return Vec3b(128, 128, 128);
}

int medianOfChannel(Mat_<uchar> channel)
{
    vector<int> hist(256, 0);
    for (int i = 0; i < channel.rows; i++) {
        for (int j = 0; j < channel.cols; j++) {
            hist[channel(i, j)]++;
        }
    }
    int half = channel.rows * channel.cols / 2;
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += hist[i];
        if (sum >= half) {
            return i;
        }
    }
    return 128;
}

Mat_<uchar> adaptiveCannyChannel(Mat_<uchar> channel)
{
    Mat_<uchar> blurred;
    GaussianBlur(channel, blurred, Size(5, 5), 1.2);
    int med = medianOfChannel(blurred);
    int low = max(12, (int)(0.66f * med));
    int high = min(245, max(low + 35, (int)(1.33f * med)));
    Mat_<uchar> edges;
    Canny(blurred, edges, low, high);
    return edges;
}

void buildAdaptiveRubikEdges(Mat_<Vec3b> img, Mat_<uchar> S, Mat_<uchar> V, Mat_<uchar>& gray, Mat_<uchar>& edgeMap, Mat_<uchar>& closedMap)
{
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat_<uchar> edgesV = adaptiveCannyChannel(V);
    Scalar meanS = mean(S);
    if (meanS[0] < 75.0) {
        edgeMap = edgesV.clone();
    }
    else {
        Mat_<uchar> edgesS = adaptiveCannyChannel(S);
        Mat_<uchar> edgesGray = adaptiveCannyChannel(gray);
        Mat_<uchar> combined;
        addWeighted(S, 0.45, V, 0.35, 0.0, combined);
        addWeighted(combined, 1.0, gray, 0.20, 0.0, combined);
        Mat_<uchar> edgesCombined = adaptiveCannyChannel(combined);
        bitwise_or(edgesV, edgesS, edgeMap);
        bitwise_or(edgeMap, edgesGray, edgeMap);
        bitwise_or(edgeMap, edgesCombined, edgeMap);
    }
    Mat kernelDilate = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat kernelClose = getStructuringElement(MORPH_RECT, Size(9, 9));
    dilate(edgeMap, closedMap, kernelDilate, Point(-1, -1), 1);
    morphologyEx(closedMap, closedMap, MORPH_CLOSE, kernelClose, Point(-1, -1), 2);
}

string classifyRubikHSVValue(int H, int S, int V)
{
    if (V < 50) {
        return "UNKNOWN";
    }
    if (S <= 48 && V >= 140) {
        return "WHITE";
    }
    if (S < 45 || V < 55) {
        return "UNKNOWN";
    }
    if (H <= 10 || H >= 244) {
        return "RED";
    }
    if (H >= 11 && H <= 32) {
        return "ORANGE";
    }
    if (H >= 33 && H <= 62) {
        return "YELLOW";
    }
    if (H >= 59 && H <= 150) {
        return "GREEN";
    }
    if (H >= 151 && H <= 235) {
        return "BLUE";
    }
    return "UNKNOWN";
}

string classifyStickerPixel(Vec3b bgr, Vec3b hsv)
{
    int B = bgr[0];
    int G = bgr[1];
    int R = bgr[2];
    int H = hsv[0];
    int S = hsv[1];
    int V = hsv[2];
    if (V < 55) {
        return "UNKNOWN";
    }
    if (V > 242 && S < 35) {
        return "UNKNOWN";
    }
    if (S <= 48 && V >= 142) {
        int maxRGB = max(R, max(G, B));
        int minRGB = min(R, min(G, B));
        if (maxRGB - minRGB <= 58) {
            return "WHITE";
        }
        return "UNKNOWN";
    }
    if (S < 48) {
        return "UNKNOWN";
    }
    if ((H <= 10 || H >= 244) && R > G + 22 && R > B + 20) {
        return "RED";
    }
    if (H >= 11 && H <= 32 && R > G + 5 && G > B + 8) {
        float gRatio = R > 0 ? (float)G / (float)R : 0.0f;
        if (gRatio >= 0.32f && gRatio <= 0.86f) {
            return "ORANGE";
        }
    }
    if (H >= 33 && H <= 62 && R > B + 25 && G > B + 25) {
        return "YELLOW";
    }
    if (H >= 63 && H <= 150 && G > R + 5 && G > B - 3) {
        return "GREEN";
    }
    if (H >= 151 && H <= 235 && B > R + 8 && B >= G - 12) {
        return "BLUE";
    }
    return classifyRubikHSVValue(H, S, V);
}

string classifyStickerCentralRegion(Mat_<Vec3b> warpedFace, Mat_<Vec3b> warpedHSV, Mat_<uchar> warpedEdges, Rect cellBox, float& confidence)
{
    Rect sample(
        cellBox.x + (int)(cellBox.width * 0.27f),
        cellBox.y + (int)(cellBox.height * 0.27f),
        max(1, (int)(cellBox.width * 0.46f)),
        max(1, (int)(cellBox.height * 0.46f))
    );
    sample = sample & Rect(0, 0, warpedHSV.cols, warpedHSV.rows);
    vector<int> votes(6, 0);
    vector<int> hs, ss, vs;
    int usable = 0;
    int whitePixels = 0;
    for (int y = sample.y; y < sample.y + sample.height; y++) {
        for (int x = sample.x; x < sample.x + sample.width; x++) {
            if (warpedEdges(y, x) != 0) {
                continue;
            }
            Vec3b hsv = warpedHSV(y, x);
            Vec3b bgr = warpedFace(y, x);
            int h = hsv[0], s = hsv[1], v = hsv[2];
            if (v < 55 || (v > 242 && s < 35)) {
                continue;
            }
            hs.push_back(h);
            ss.push_back(s);
            vs.push_back(v);
            string color = classifyStickerPixel(bgr, hsv);
            if (color == "WHITE") {
                votes[0]++;
                whitePixels++;
            }
            else if (color == "YELLOW") votes[1]++;
            else if (color == "RED") votes[2]++;
            else if (color == "ORANGE") votes[3]++;
            else if (color == "GREEN") votes[4]++;
            else if (color == "BLUE") votes[5]++;
            usable++;
        }
    }
    if (usable == 0) {
        confidence = 0.0f;
        return "UNKNOWN";
    }
    sort(hs.begin(), hs.end());
    sort(ss.begin(), ss.end());
    sort(vs.begin(), vs.end());
    int medianH = hs[hs.size() / 2];
    int medianS = ss[ss.size() / 2];
    int medianV = vs[vs.size() / 2];
    float whiteRatio = (float)whitePixels / (float)usable;
    if (whiteRatio >= 0.58f && medianS <= 58 && medianV >= 135) {
        confidence = whiteRatio;
        return "WHITE";
    }
    int bestIndex = 0;
    for (int i = 1; i < 6; i++) {
        if (votes[i] > votes[bestIndex]) {
            bestIndex = i;
        }
    }
    confidence = (float)votes[bestIndex] / (float)usable;
    const string names[6] = {"WHITE", "YELLOW", "RED", "ORANGE", "GREEN", "BLUE"};
    if (bestIndex == 0 && confidence < 0.58f) {
        votes[0] = 0;
        bestIndex = 1;
        for (int i = 2; i < 6; i++) {
            if (votes[i] > votes[bestIndex]) {
                bestIndex = i;
            }
        }
        confidence = (float)votes[bestIndex] / (float)usable;
    }
    if (votes[bestIndex] > 0 && confidence >= 0.34f) {
        return names[bestIndex];
    }
    confidence = max(confidence, 0.25f);
    return classifyRubikHSVValue(medianH, medianS, medianV);
}

Mat_<uchar> buildWarpedStickerRegionMask(Mat_<Vec3b> warpedHSV)
{
    Mat_<uchar> mask(warpedHSV.rows, warpedHSV.cols);
    mask.setTo(0);
    for (int y = 0; y < warpedHSV.rows; y++) {
        for (int x = 0; x < warpedHSV.cols; x++) {
            Vec3b hsv = warpedHSV(y, x);
            int s = hsv[1];
            int v = hsv[2];
            bool whiteSticker = s <= 78 && v >= 120;
            bool coloredSticker = s >= 36 && v >= 55;
            if (whiteSticker || coloredSticker) {
                mask(y, x) = 255;
            }
        }
    }
    Mat kernelOpen = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat kernelClose = getStructuringElement(MORPH_RECT, Size(9, 9));
    morphologyEx(mask, mask, MORPH_OPEN, kernelOpen, Point(-1, -1), 1);
    morphologyEx(mask, mask, MORPH_CLOSE, kernelClose, Point(-1, -1), 1);
    return mask;
}

vector<StickerInfo> detectWarpedStickerContours(Mat_<uchar> stickerMask)
{
    vector<StickerInfo> candidates;
    vector<vector<Point>> contours;
    findContours(stickerMask.clone(), contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    const float cellArea = 100.0f * 100.0f;
    for (const vector<Point>& contour : contours) {
        double area = contourArea(contour);
        if (area < cellArea * 0.12 || area > cellArea * 1.25) {
            continue;
        }
        RotatedRect rr = minAreaRect(contour);
        if (rr.size.width < 22 || rr.size.height < 22) {
            continue;
        }
        float aspect = max(rr.size.width, rr.size.height) / max(1.0f, min(rr.size.width, rr.size.height));
        if (aspect > 1.65f) {
            continue;
        }
        float rectangularity = (float)area / max(1.0f, rr.size.width * rr.size.height);
        if (rectangularity < 0.28f) {
            continue;
        }
        Point2f c = rr.center;
        int col = min(2, max(0, (int)(c.x / 100.0f)));
        int row = min(2, max(0, (int)(c.y / 100.0f)));
        Point2f expected(50.0f + col * 100.0f, 50.0f + row * 100.0f);
        float dist = (float)norm(c - expected);
        if (dist > 44.0f) {
            continue;
        }
        StickerInfo sticker;
        sticker.rotatedBox = rr;
        sticker.box = boundingRect(contour);
        sticker.center = c;
        sticker.row = row;
        sticker.col = col;
        sticker.confidence = max(0.0f, 1.0f - dist / 70.0f);
        candidates.push_back(sticker);
    }
    return candidates;
}

Mat_<uchar> buildStickerContourMap(Mat_<Vec3b> warpedHSV, Mat_<uchar> warpedEdges, Mat_<uchar> warpedClosed)
{
    Mat_<uchar> stickerMask = buildWarpedStickerRegionMask(warpedHSV);
    Mat_<uchar> contourMap;
    bitwise_or(stickerMask, warpedClosed, contourMap);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(contourMap, contourMap, MORPH_CLOSE, kernel, Point(-1, -1), 1);
    bitwise_or(contourMap, warpedEdges, contourMap);
    return contourMap;
}

Mat_<Vec3b> drawWarpedStickerCandidates(Mat_<Vec3b> warpedFace, const vector<StickerInfo>& candidates)
{
    Mat_<Vec3b> debug = warpedFace.clone();
    for (const StickerInfo& sticker : candidates) {
        Point2f pts[4];
        sticker.rotatedBox.points(pts);
        for (int i = 0; i < 4; i++) {
            line(debug, pts[i], pts[(i + 1) % 4], Scalar(0, 255, 0), 2);
        }
        circle(debug, sticker.center, 2, Scalar(0, 0, 255), FILLED);
    }
    return debug;
}

StickerInfo mapStickerToOriginal(const StickerInfo& warpedSticker, Mat inversePerspective)
{
    Point2f srcPts[4];
    if (warpedSticker.rotatedBox.size.width > 1 && warpedSticker.rotatedBox.size.height > 1) {
        warpedSticker.rotatedBox.points(srcPts);
    }
    else {
        srcPts[0] = Point2f((float)warpedSticker.box.x, (float)warpedSticker.box.y);
        srcPts[1] = Point2f((float)(warpedSticker.box.x + warpedSticker.box.width), (float)warpedSticker.box.y);
        srcPts[2] = Point2f(
            (float)(warpedSticker.box.x + warpedSticker.box.width),
            (float)(warpedSticker.box.y + warpedSticker.box.height)
        );
        srcPts[3] = Point2f((float)warpedSticker.box.x, (float)(warpedSticker.box.y + warpedSticker.box.height));
    }
    vector<Point2f> src(srcPts, srcPts + 4), dst;
    perspectiveTransform(src, dst, inversePerspective);
    StickerInfo mapped = warpedSticker;
    mapped.center = Point2f(0, 0);
    for (const Point2f& p : dst) {
        mapped.center += p;
    }
    mapped.center *= 0.25f;
    mapped.rotatedBox = minAreaRect(dst);
    vector<Point> intPts;
    for (const Point2f& p : dst) {
        intPts.push_back(Point(cvRound(p.x), cvRound(p.y)));
    }
    mapped.box = boundingRect(intPts);
    return mapped;
}

vector<StickerInfo> extractRubikStickers(Mat_<Vec3b> img, Mat_<uchar> H, Mat_<uchar> S, Mat_<uchar> V,const RubikFace& face, Mat_<Vec3b>& warpedFace,Mat_<uchar>& warpedEdges, Mat_<uchar>& warpedClosed,Mat_<Vec3b>& warpedStickerCandidates)
{
    vector<StickerInfo> finalStickers;
    if (!face.valid) {
        return finalStickers;
    }
    vector<Point2f> dstQuad;
    dstQuad.push_back(Point2f(0, 0));
    dstQuad.push_back(Point2f(299, 0));
    dstQuad.push_back(Point2f(299, 299));
    dstQuad.push_back(Point2f(0, 299));
    Mat perspective = getPerspectiveTransform(face.corners, dstQuad);
    Mat inversePerspective = getPerspectiveTransform(dstQuad, face.corners);
    warpPerspective(img, warpedFace, perspective, Size(300, 300), INTER_LINEAR, BORDER_REPLICATE);
    Mat_<Vec3b> warpedHSV = convertRGBtoHSV(warpedFace);
    Mat_<uchar> wh, ws, wv;
    splitHSVChannels(warpedHSV, wh, ws, wv);
    Mat_<uchar> warpedGray;
    buildAdaptiveRubikEdges(warpedFace, ws, wv, warpedGray, warpedEdges, warpedClosed);
    Mat_<uchar> stickerContourMap = buildStickerContourMap(warpedHSV, warpedEdges, warpedClosed);
    vector<StickerInfo> contourStickers = detectWarpedStickerContours(stickerContourMap);
    warpedStickerCandidates = drawWarpedStickerCandidates(warpedFace, contourStickers);
    StickerInfo bestByCell[3][3];
    bool hasContourCell[3][3] = {};
    for (StickerInfo sticker : contourStickers) {
        if (sticker.row < 0 || sticker.row >= 3 || sticker.col < 0 || sticker.col >= 3) {
            continue;
        }
        if (!hasContourCell[sticker.row][sticker.col] ||
            sticker.confidence > bestByCell[sticker.row][sticker.col].confidence) {
            bestByCell[sticker.row][sticker.col] = sticker;
            hasContourCell[sticker.row][sticker.col] = true;
        }
    }
    int contourCells = 0;
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            if (hasContourCell[row][col]) {
                contourCells++;
            }
        }
    }
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            StickerInfo warpedSticker;
            if (contourCells >= 9 && hasContourCell[row][col]) {
                warpedSticker = bestByCell[row][col];
            }
            else {
                warpedSticker.row = row;
                warpedSticker.col = col;
                warpedSticker.box = Rect(col * 100, row * 100, 100, 100);
                warpedSticker.center = Point2f(50.0f + col * 100.0f, 50.0f + row * 100.0f);
                warpedSticker.rotatedBox = RotatedRect(warpedSticker.center, Size2f(98, 98), 0.0f);
                warpedSticker.confidence = 0.35f;
            }
            Rect cellBox(col * 100, row * 100, 100, 100);
            float colorConfidence = 0.0f;
            warpedSticker.color = classifyStickerCentralRegion(warpedFace, warpedHSV, warpedEdges, cellBox, colorConfidence);
            warpedSticker.confidence = max(warpedSticker.confidence, colorConfidence);
            finalStickers.push_back(mapStickerToOriginal(warpedSticker, inversePerspective));
        }
    }
    return finalStickers;
}

void applyFaceColorConsistency(vector<StickerInfo>& stickers)
{
    vector<string> colors = {"WHITE", "YELLOW", "RED", "ORANGE", "GREEN", "BLUE"};
    vector<int> counts(colors.size(), 0);
    for (const StickerInfo& sticker : stickers) {
        for (size_t i = 0; i < colors.size(); i++) {
            if (sticker.color == colors[i]) {
                counts[i]++;
            }
        }
    }
    int best = 0;
    for (int i = 1; i < (int)counts.size(); i++) {
        if (counts[i] > counts[best]) {
            best = i;
        }
    }
    if (counts[best] < 7) {
        return;
    }
    string dominant = colors[best];
    for (StickerInfo& sticker : stickers) {
        if (dominant != "WHITE" && sticker.color == "WHITE" && sticker.confidence < 0.75f) {
            sticker.color = dominant;
            sticker.confidence = min(0.70f, sticker.confidence + 0.15f);
        }
        if (dominant == "RED" && sticker.color == "ORANGE" && sticker.confidence < 0.55f) {
            sticker.color = dominant;
        }
        if (dominant == "ORANGE" && sticker.color == "RED" && sticker.confidence < 0.55f) {
            sticker.color = dominant;
        }
    }
}

Mat_<Vec3b> drawFaceAndStickerGrid(Mat_<Vec3b> img, const RubikFace& face, const vector<StickerInfo>& stickers, bool printMatrix)
{
    Mat_<Vec3b> result = img.clone();
    if (face.valid && face.corners.size() == 4) {
        vector<Point> facePts;
        for (const Point2f& p : face.corners) {
            facePts.push_back(Point(cvRound(p.x), cvRound(p.y)));
        }
        polylines(result, facePts, true, Scalar(0, 0, 255), 3);
    }
    string matrix[3][3];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            matrix[r][c] = "UNKNOWN";
        }
    }
    for (const StickerInfo& sticker : stickers) {
        Vec3b bgr = colorNameToBGR(sticker.color);
        Point2f pts[4];
        sticker.rotatedBox.points(pts);
        for (int i = 0; i < 4; i++) {
            line(result, pts[i], pts[(i + 1) % 4], Scalar(bgr[0], bgr[1], bgr[2]), 2);
        }
        circle(result, sticker.center, 3, Scalar(255, 255, 255), FILLED);
        string label = to_string(sticker.row) + "," + to_string(sticker.col) + " " + sticker.color;
        putText(result, label, sticker.center + Point2f(-32, 4), FONT_HERSHEY_SIMPLEX, 0.38, Scalar(0, 0, 0), 3);
        putText(result, label, sticker.center + Point2f(-32, 4), FONT_HERSHEY_SIMPLEX, 0.38, Scalar(255, 255, 255), 1);
        if (sticker.row >= 0 && sticker.row < 3 && sticker.col >= 0 && sticker.col < 3) {
            matrix[sticker.row][sticker.col] = sticker.color;
        }
    }
    if (printMatrix) {
        cout << "\nDetected Rubik stickers:" << endl;
        for (const StickerInfo& sticker : stickers) {
            cout << "  [" << sticker.row << "," << sticker.col << "] = " << sticker.color
                 << " confidence=" << fixed << setprecision(2) << sticker.confidence << endl;
        }
        cout << "\nRubik 3x3 color matrix:" << endl;
        for (int r = 0; r < 3; r++) {
            cout << "  ";
            for (int c = 0; c < 3; c++) {
                cout << matrix[r][c];
                if (c < 2) {
                    cout << " | ";
                }
            }
            cout << endl;
        }
    }
    return result;
}

void printComponents(string title, vector<ComponentInfo> components)
{
    cout << "\n" << title << ": " << components.size() << endl;
    int limit = min((int)components.size(), 20);
    for (int i = 0; i < limit; i++) {
        ComponentInfo c = components[i];
        cout << i + 1
             << " label=" << c.label
             << " area=" << c.area
             << " center=(" << c.center.x << "," << c.center.y << ")"
             << " bbox=" << c.bbox.width << "x" << c.bbox.height
             << " aspect=" << c.aspect
             << " density=" << c.bboxDensity
             << " HSV=(" << c.meanH << "," << c.meanS << "," << c.meanV << ")"
             << " RGB=(" << c.meanR << "," << c.meanG << "," << c.meanB << ")"
             << " color=" << c.colorName << endl;
    }
}

void printFace(ComponentInfo face)
{
    cout << "\nBest face candidate:" << endl;
    if (face.area == 0) {
        cout << "No face found." << endl;
        return;
    }
    cout << "label=" << face.label
         << " area=" << face.area
         << " center=(" << face.center.x << "," << face.center.y << ")"
         << " bbox=" << face.bbox.width << "x" << face.bbox.height
         << " aspect=" << face.aspect
         << " density=" << face.bboxDensity
         << " HSV=(" << face.meanH << "," << face.meanS << "," << face.meanV << ")"
         << " RGB=(" << face.meanR << "," << face.meanG << "," << face.meanB << ")"
         << " color=" << face.colorName << endl;
}

RubikFace faceFromComponent(ComponentInfo component, Size imageSize)
{
    RubikFace face;
    if (component.area == 0) {
        return face;
    }
    Rect box = component.bbox & Rect(0, 0, imageSize.width, imageSize.height);
    if (box.width <= 0 || box.height <= 0) {
        return face;
    }
    face.corners.push_back(Point2f((float)box.x, (float)box.y));
    face.corners.push_back(Point2f((float)(box.x + box.width - 1), (float)box.y));
    face.corners.push_back(Point2f((float)(box.x + box.width - 1), (float)(box.y + box.height - 1)));
    face.corners.push_back(Point2f((float)box.x, (float)(box.y + box.height - 1)));
    face.bbox = box;
    face.valid = true;
    return face;
}
ComponentInfo detectRubikFaceROI(Mat_<Vec3b> img, Mat_<uchar>& H, Mat_<uchar>& S, Mat_<uchar>& V,Mat_<uchar>& coloredMask, Mat_<uchar>& cleanColoredMask,Mat_<int>& coloredLabels, vector<ComponentInfo>& coloredComponents,bool showDebugWindows, bool& usedWhiteCanny)
{
    coloredMask = buildColoredRubikMask(img, H, S, V);
    Mat_<uchar> strel3 = squareStrel3();
    Mat_<uchar> strel5 = squareStrel5();
    cleanColoredMask = openingOp(coloredMask, strel3);
    cleanColoredMask = closingOp(cleanColoredMask, strel5);
    coloredLabels = Mat_<int>(cleanColoredMask.rows, cleanColoredMask.cols, 0);
    bfs_connected_components(cleanColoredMask, coloredLabels, true);
    coloredComponents = extractComponentsFromLabels(coloredLabels, H, S, V, img);
    ComponentInfo face = findBestComponentFace(coloredComponents, img.size());
    usedWhiteCanny = false;
    if (face.area == 0) {
        face = findWhiteFaceWithCanny(img, H, S, V, showDebugWindows);
        usedWhiteCanny = true;
    }
    if (face.area > 0 && face.colorName != "WHITE") {
        face.colorName = dominantColorInsideBox(img, H, S, V, face.bbox);
    }
    return face;
}

Mat_<Vec3b> processRubikFrame(Mat_<Vec3b> inputImg, bool showDebugWindows, bool printInfo)
{
    Mat_<Vec3b> img = resizeProjectImageIfNeeded(inputImg);
    if (img.empty()) {
        return img;
    }
    Mat_<uchar> H, S, V;
    Mat_<uchar> coloredMask;
    Mat_<uchar> cleanColoredMask;
    Mat_<int> coloredLabels;
    vector<ComponentInfo> coloredComponents;
    bool usedWhiteCanny = false;
    ComponentInfo faceComponent = detectRubikFaceROI(
        img, H, S, V, coloredMask, cleanColoredMask, coloredLabels, coloredComponents,
        showDebugWindows, usedWhiteCanny
    );
    RubikFace face = faceFromComponent(faceComponent, img.size());
    Mat_<Vec3b> warpedFace;
    Mat_<uchar> warpedEdges;
    Mat_<uchar> warpedClosed;
    Mat_<Vec3b> warpedStickerCandidates;
    vector<StickerInfo> finalStickers;
    if (face.valid) {
        finalStickers = extractRubikStickers(img, H, S, V, face, warpedFace, warpedEdges, warpedClosed, warpedStickerCandidates);
        applyFaceColorConsistency(finalStickers);
    }
    Mat_<Vec3b> result = drawFaceAndStickerGrid(img, face, finalStickers, printInfo);
    if (showDebugWindows) {
        imshow("1 Original resized", img);
        imshow("2 H channel", H);
        imshow("3 S channel", S);
        imshow("4 V channel", V);
        if (!usedWhiteCanny) {
            imshow("5 Colored mask", coloredMask);
            imshow("6 Clean colored mask", cleanColoredMask);
            displayLabels(coloredLabels, "7 Colored connected components");
        }
        if (!warpedFace.empty()) {
            imshow("8 ROI warped face", warpedFace);
        }
        if (!warpedEdges.empty()) {
            imshow("9 ROI adaptive edges", warpedEdges);
        }
        if (!warpedClosed.empty()) {
            imshow("10 ROI closed edges", warpedClosed);
        }
        if (!warpedStickerCandidates.empty()) {
            imshow("11 ROI sticker candidates", warpedStickerCandidates);
        }
        imshow("12 Final Rubik grid", result);
    }
    if (printInfo) {
        if (!usedWhiteCanny) {
            printComponents("Colored components", coloredComponents);
        }
        printFace(faceComponent);
        cout << "Stickers inside ROI: " << finalStickers.size() << endl;
    }
    return result;
}

bool tryOpenCamera(VideoCapture& cap)
{
    vector<int> cameraIndexes = {0, 1, 2, 3};
    vector<int> backends = {CAP_AVFOUNDATION, CAP_ANY};
    for (int backend : backends) {
        for (int index : cameraIndexes) {
            cout << "Trying camera index " << index << " with backend " << backend << endl;
            cap.open(index, backend);
            if (!cap.isOpened()) {
                cap.release();
                continue;
            }
            cap.set(CAP_PROP_FRAME_WIDTH, 640);
            cap.set(CAP_PROP_FRAME_HEIGHT, 480);
            cap.set(CAP_PROP_FPS, 30);
            Mat frame;
            for (int k = 0; k < 30; k++) {
                cap.read(frame);
                waitKey(30);
                if (!frame.empty()) {
                    cout << "Camera works with index " << index << endl;
                    return true;
                }
            }
            cout << "Camera opened, but no valid frames." << endl;
            cap.release();
        }
    }
    return false;
}

bool readValidFrame(VideoCapture& cap, Mat& frame)
{
    for (int k = 0; k < 10; k++) {
        cap.read(frame);
        if (!frame.empty()) {
            return true;
        }
        waitKey(20);
    }
    return false;
}

void project()
{
    int op;
    do {
        cout << "\nPROJECT MENU:" << endl;
        cout << "1 - Run on images from Project/ folder" << endl;
        cout << "2 - Open camera and detect live" << endl;
        cout << "0 - Exit project" << endl;
        cout << "Option: ";
        cin >> op;
        switch (op) {
            case 1:
            {
                vector<string> filenames = {
                    "Project/portocaliu.bmp",
                    "Project/rosu.bmp",
                    "Project/galben.bmp",
                    "Project/verde.bmp",
                    "Project/albastru.bmp",
                    "Project/alb.bmp"
                };
                for (const string& filename : filenames) {
                    cout << filename << endl;
                    Mat_<Vec3b> img = imread(filename);
                    Mat_<Vec3b> result = processRubikFrame(img, true, true);
                    imshow("8 Detected face and simple 3x3 grid", result);
                    waitKey(0);
                    destroyAllWindows();
                }
                break;
            }
            case 2:
            {
                VideoCapture cap;
                if (!tryOpenCamera(cap)) {
                    cout << "Could not open a working camera" << endl;
                    break;
                }
                while (true) {
                    Mat frame;
                    if (!readValidFrame(cap, frame)) {
                        cout << "Could not read valid frame from camera" << endl;
                        break;
                    }
                    Mat_<Vec3b> img = frame;
                    Mat_<Vec3b> result = processRubikFrame(img, false, false);
                    imshow("Live Rubik detection", result);
                    char key = (char)waitKey(1);
                    if (key == 27 || key == 'q' || key == 'Q') {
                        break;
                    }
                }
                cap.release();
                destroyAllWindows();
                break;
            }
            case 0:
            {
                cout << "Exit project" << endl;
                break;
            }
            default:
            {
                cout << "Invalid option" << endl;
                break;
            }
        }
    } while (op != 0);
}

int main(){
    int op;
    do{
        printf("Menu:\n");
        printf(" 1 - Lab1 \n");
        printf(" 2 - Lab2 \n");
        printf(" 3 - Lab3 \n");
        printf(" 4 - Lab4 \n");
        printf(" 5 - Lab5 \n");
        printf(" 6 - Lab6 \n");
        printf(" 7 - Lab7 \n");
        printf(" 8 - Lab8 \n");
        printf(" 9 - Lab9 \n");
        printf(" 10 - Lab10 \n");
        printf(" 11 - Lab11 \n");
        printf(" 12 - Lab12 \n");
        printf(" 15 - Project \n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d",&op);
        switch (op)
        {
            case 1:
                lab1_main();
                break;
            case 2:
                lab2();
                break;
            case 3:
                lab3();
                break;
            case 4:
                lab4();
                break;
            case 5:
                lab5();
                break;
            case 6:
                lab6();
                break;
            case 7:
                lab7();
                break;
            case 8:
                lab8();
                break;
            case 9:
                lab9();
                break;
            case 10:
                lab10();
                break;
            case 11:
                lab11();
                break;
            case 12:
                lab12();
                break;
            case 15:
                project();
                break;
        }
    }
    while (op!=0);
    return 0;
}
