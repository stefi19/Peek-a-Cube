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

Mat_<uchar> applyGaussianForCanny(Mat_<uchar> img)
{
    int w = 3; //sigma = w/6
    Mat_<float> Gx = createGaussianKernelGx(w);
    Mat_<float> Gy = createGaussianKernelGy(w);
    Mat_<uchar> result = applyGaussian1D(img, w, Gx, Gy);
    return result;
}

Mat_<float> sobelX(Mat_<uchar> img)
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
            float sum = 0;
            for (int u = 0; u < 3; u++)
            {
                for (int v = 0; v < 3; v++)
                {
                    int x = i + u - 1;
                    int y = j + v - 1;
                    sum = sum + img(x, y) * kernel[u][v];
                }
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Mat_<float> sobelY(Mat_<uchar> img)
{
    Mat_<float> result(img.rows, img.cols);
    result.setTo(0);
    int kernel[3][3] = {
        {1, 2, 1},
        { 0,  0,  0},
        { -1,  -2,  -1}
    };
    for (int i = 1; i < img.rows - 1; i++)
    {
        for (int j = 1; j < img.cols - 1; j++)
        {
            float sum = 0;
            for (int u = 0; u < 3; u++)
            {
                for (int v = 0; v < 3; v++)
                {
                    int x = i + u - 1;
                    int y = j + v - 1;
                    sum = sum + img(x, y) * kernel[u][v];
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
    for (int i = 0; i < direction.rows; i++)
    {
        for (int j = 0; j < direction.cols; j++)
        {
            float phi = direction(i, j);
            if (phi < 0)
            {
                phi = phi + 2 * CV_PI;
            }
            int value = (int)(phi * 8.0f / (2.0f * CV_PI) + 0.5f);
            q(i, j) = value % 8;
        }
    }
    return q;
}

Mat_<float> nonMaximaSuppression(Mat_<float> magnitude, Mat_<uchar> q)
{
    Mat_<float> thinnedMag(magnitude.rows, magnitude.cols);
    thinnedMag.setTo(0);
    int di[8] = {0, -1, -1, -1, 0, 1, 1, 1};
    int dj[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    for (int i = 1; i < magnitude.rows - 1; i++)
    {
        for (int j = 1; j < magnitude.cols - 1; j++)
        {
            int direction = q(i, j);
            int i1 = i + di[direction];
            int j1 = j + dj[direction];
            int i2 = i - di[direction];
            int j2 = j - dj[direction];

            float current = magnitude(i, j);
            float neighbor1 = magnitude(i1, j1);
            float neighbor2 = magnitude(i2, j2);

            if (current >= neighbor1 && current >= neighbor2)
            {
                thinnedMag(i, j) = current;
            }
            else
            {
                thinnedMag(i, j) = 0;
            }
        }
    }
    return thinnedMag;
}
Mat_<uchar> scaleSobelMagnitude(Mat_<float> mag)
{
    Mat_<uchar> result(mag.rows, mag.cols);
    result.setTo(0);
    float scale = 4.0f * sqrt(2.0f);
    for (int i = 0; i < mag.rows; i++)
    {
        for (int j = 0; j < mag.cols; j++)
        {
            int value = (int)(mag(i, j) / scale);
            if (value > 255)
            {
                value = 255;
            }
            if (value < 0)
            {
                value = 0;
            }
            result(i, j) = (uchar)value;
        }
    }
    return result;
}
Mat_<float> adaptiveThresholdResult(Mat_<float> thinnedMag, float threshold)
{
    Mat_<float> result(thinnedMag.rows, thinnedMag.cols);
    result.setTo(0);
    for (int i = 0; i < thinnedMag.rows; i++)
    {
        for (int j = 0; j < thinnedMag.cols; j++)
        {
            if (thinnedMag(i, j) >= threshold)
            {
                result(i, j) = 255.0f;
            }
            else
            {
                result(i, j) = 0.0f;
            }
        }
    }
    return result;
}
float adaptiveThresholding(Mat_<float> thinnedMag, float p)
{
    vector<int> hist(256, 0);
    float scale = 4.0f * sqrt(2.0f);
    for (int i = 0; i < thinnedMag.rows; i++)
    {
        for (int j = 0; j < thinnedMag.cols; j++)
        {
            int value = (int)(thinnedMag(i, j) / scale);
            if (value > 255)
            {
                value = 255;
            }
            if (value < 0)
            {
                value = 0;
            }
            hist[value]++;
        }
    }
    int noNonEdge = (int)((1.0-p) * (thinnedMag.rows * thinnedMag.cols - hist[0]));
    int sum = 0;
    int thresholdScaled = 0;
    for (int i = 1; i < 256; i++)
    {
        sum = sum + hist[i];
        if (sum > noNonEdge)
        {
            thresholdScaled = i;
            break;
        }
    }
    float threshold = thresholdScaled * scale;
    return threshold;
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
            if (thinnedMag(i, j) >= thresholdHigh)
            {
                result(i, j) = 255.0f; // strong edge
            }
            else if (thinnedMag(i, j) >= thresholdLow)
            {
                result(i, j) = 128.0f; // weak edge
            }
            else
            {
                result(i, j) = 0.0f; // non edge
            }
        }
    }
    return result;
}

Mat_<float> edgeExtensionHysteresis(Mat_<float> edges)
{
    Mat_<float> result = edges.clone();
    queue<Point> Q;
    int di[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    int dj[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            if (result(i, j) == 255.0f)
            {
                Q.push(Point(j, i));
            }
        }
    }
    while (!Q.empty())
    {
        Point current = Q.front();
        Q.pop();
        for (int d = 0; d < 8; d++)
        {
            int ni = current.y + di[d];
            int nj = current.x + dj[d];
            if (isInside(result, ni, nj) && result(ni, nj) == 128.0f)
            {
                result(ni, nj) = 255.0f;
                Q.push(Point(nj, ni));
            }
        }
    }
    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            if (result(i, j) == 128.0f)
            {
                result(i, j) = 0.0f;
            }
        }
    }
    return result;
}
void cannyGradientStep()
{
    Mat_<uchar> img = imread("PI-L11/cameraman.bmp", IMREAD_GRAYSCALE);

    Mat_<uchar> gaussian = applyGaussianForCanny(img);

    Mat_<float> dx = sobelX(gaussian);
    Mat_<float> dy = sobelY(gaussian);

    Mat_<float> magnitude = gradientMagnitude(dx, dy);
    Mat_<float> gradient = gradientDirection(dx, dy);

    Mat_<uchar> q = computeQ(gradient);

    Mat_<float> thinnedMag = nonMaximaSuppression(magnitude, q);

    float p = 0.1f;
    float k = 0.4f;

    float thresholdHigh = adaptiveThresholding(thinnedMag, p);

    cout << "Threshold high: " << thresholdHigh << endl;
    cout << "Threshold low: " << k * thresholdHigh << endl;
    float scale = 4.0f * sqrt(2.0f);
    cout << "Threshold high scaled: " << thresholdHigh / scale << endl;
    cout << "Threshold low scaled: " << (k * thresholdHigh) / scale << endl;

    Mat_<float> hysteresisImage = hysteresisThresholding(thinnedMag, thresholdHigh, k);
    Mat_<float> finalEdges = edgeExtensionHysteresis(hysteresisImage);

    imshow("Initial image", img);
    imshow("Gaussian image", gaussian);
    imshow("dx", dx / 255);
    imshow("dy", dy / 255);
    imshow("thinnedMag", thinnedMag / 255);
    imshow("Strong and weak edges", hysteresisImage / 255);
    imshow("Final Canny edges", finalEdges / 255);

    waitKey(0);
}

void lab12() {
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
    int perimeter = 0;

    Point2f center = Point2f(0, 0);
    Rect bbox;

    float aspect = 0.0f;
    float thinness = 0.0f;

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

bool isSkinLikeHSV(int H, int S, int V)
{
    bool skinHue = (H >= 0 && H <= 35);
    bool skinSaturation = (S >= 18 && S <= 135);
    bool skinValue = (V >= 65);

    return skinHue && skinSaturation && skinValue;
}

bool isWarmSkinWhitePixel(int H, int S, int V, int R, int G, int B)
{
    bool lowSatWarm =
        S <= 85 &&
        V >= 90 &&
        H <= 38 &&
        R > G + 4 &&
        R > B + 10 &&
        G >= B - 5;

    return lowSatWarm;
}

bool isWhiteRubikPixel(int H, int S, int V, int R, int G, int B)
{
    // S important
    // alb = saturatie mica + luminozitate mare + RGB echilibrat.
    if (isWarmSkinWhitePixel(H, S, V, R, G, B)) {
        return false;
    }

    bool lowSaturation = S <= 62;

    bool brightEnough =
        V >= 118 &&
        R >= 108 &&
        G >= 108 &&
        B >= 108;

    bool balancedRGB =
        abs(R - G) <= 48 &&
        abs(R - B) <= 58 &&
        abs(G - B) <= 58;

    return lowSaturation && brightEnough && balancedRGB;
}

string classifyRubikColor(int H, int S, int V, int R, int G, int B)
{
    if (isWhiteRubikPixel(H, S, V, R, G, B)) {
        return "WHITE";
    }

    float gRatio = 0.0f;
    float bRatio = 0.0f;

    if (R > 0) {
        gRatio = (float)G / (float)R;
        bRatio = (float)B / (float)R;
    }

    bool red =
        R >= 115 &&
        R > G + 40 &&
        R > B + 35 &&
        gRatio <= 0.43f &&
        (H <= 16 || H >= 235);

    if (red) {
        return "RED";
    }

    bool orange =
        R >= 115 &&
        G >= 45 &&
        B <= 165 &&
        gRatio > 0.43f &&
        gRatio <= 0.95f &&
        bRatio <= 0.80f &&
        R > B + 18 &&
        H <= 68;

    if (orange) {
        return "ORANGE";
    }

    bool yellow =
        R >= 120 &&
        G >= 110 &&
        B <= 170 &&
        abs(R - G) <= 105 &&
        H > 35 &&
        H <= 88;

    if (yellow) {
        return "YELLOW";
    }

    bool green =
        G >= 75 &&
        G > R + 5 &&
        G > B + 5 &&
        H > 58 &&
        H <= 145;

    if (green) {
        return "GREEN";
    }

    bool blue =
        B >= 70 &&
        B > G + 3 &&
        B > R + 6 &&
        H > 110 &&
        H <= 230;

    if (blue) {
        return "BLUE";
    }

    if (H <= 16 || H >= 235) return "RED";
    if (H <= 68) return "ORANGE";
    if (H <= 88) return "YELLOW";
    if (H <= 145) return "GREEN";
    if (H <= 230) return "BLUE";

    return "UNKNOWN";
}

bool isColoredRubikPixel(int H, int S, int V, int R, int G, int B)
{
    if (V < 30) {
        return false;
    }

    float gRatio = 0.0f;
    float bRatio = 0.0f;

    if (R > 0) {
        gRatio = (float)G / (float)R;
        bRatio = (float)B / (float)R;
    }

    if (isSkinLikeHSV(H, S, V)) {
        bool strongRedCube =
            S >= 120 &&
            R >= 115 &&
            R > G + 40 &&
            R > B + 35 &&
            gRatio <= 0.43f;

        bool strongOrangeCube =
            S >= 110 &&
            R >= 115 &&
            G >= 45 &&
            gRatio > 0.43f &&
            gRatio <= 0.95f &&
            R > B + 18;

        if (!strongRedCube && !strongOrangeCube) {
            return false;
        }
    }

    bool red =
        R >= 115 &&
        R > G + 40 &&
        R > B + 35 &&
        gRatio <= 0.43f &&
        S >= 80 &&
        (H <= 16 || H >= 235);

    bool orange =
        R >= 115 &&
        G >= 45 &&
        B <= 165 &&
        gRatio > 0.43f &&
        gRatio <= 0.95f &&
        bRatio <= 0.80f &&
        R > B + 18 &&
        S >= 55 &&
        H <= 68;

    bool yellow =
        R >= 120 &&
        G >= 110 &&
        B <= 170 &&
        abs(R - G) <= 105 &&
        S >= 55 &&
        H > 35 &&
        H <= 88;

    bool green =
        G >= 75 &&
        G > R + 5 &&
        G > B + 5 &&
        S >= 35 &&
        H > 58 &&
        H <= 145;

    bool blue =
        B >= 70 &&
        B > G + 3 &&
        B > R + 6 &&
        S >= 32 &&
        H > 110 &&
        H <= 230;

    return red || orange || yellow || green || blue;
}

Mat_<uchar> buildRubikMask(Mat_<Vec3b> img, Mat_<uchar>& H, Mat_<uchar>& S, Mat_<uchar>& V)
{
    Mat_<Vec3b> hsv = convertRGBtoHSV(img);

    splitHSVChannels(hsv, H, S, V);

    Mat_<uchar> mask(img.rows, img.cols);
    mask.setTo(255);

    int objectPixels = 0;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int B = img(i, j)[0];
            int G = img(i, j)[1];
            int R = img(i, j)[2];

            if (isColoredRubikPixel(H(i, j), S(i, j), V(i, j), R, G, B)) {
                mask(i, j) = 0;
                objectPixels++;
            }
        }
    }

    cout << "Colored Rubik mask object pixels = " << objectPixels << endl;

    return mask;
}

float pointDistance(Point2f a, Point2f b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

float componentDiag(ComponentInfo c)
{
    return sqrt((float)c.bbox.width * c.bbox.width + (float)c.bbox.height * c.bbox.height);
}

Rect expandRectSafe(Rect r, int margin, Size size)
{
    int x1 = max(0, r.x - margin);
    int y1 = max(0, r.y - margin);
    int x2 = min(size.width - 1, r.x + r.width + margin);
    int y2 = min(size.height - 1, r.y + r.height + margin);

    return Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);
}

Mat_<int> buildObjectIntegralImage(Mat_<uchar> mask)
{
    Mat_<int> integral(mask.rows + 1, mask.cols + 1, 0);

    for (int i = 0; i < mask.rows; i++) {
        int rowSum = 0;

        for (int j = 0; j < mask.cols; j++) {
            if (mask(i, j) == 0) {
                rowSum++;
            }

            integral(i + 1, j + 1) = integral(i, j + 1) + rowSum;
        }
    }

    return integral;
}

int getObjectCountInRect(Mat_<int> integral, Rect r)
{
    int x1 = r.x;
    int y1 = r.y;
    int x2 = r.x + r.width;
    int y2 = r.y + r.height;

    return integral(y2, x2) - integral(y1, x2) - integral(y2, x1) + integral(y1, x1);
}

vector<ComponentInfo> extractComponentsFromLabels( Mat_<int> labels, Mat_<uchar> H, Mat_<uchar> S, Mat_<uchar> V, Mat_<Vec3b> img) {
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

    vector<int> perim(maxLabel + 1, 0);

    int di[4] = { -1, 0, 1, 0 };
    int dj[4] = { 0, -1, 0, 1 };

    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            int label = labels(i, j);

            if (label > 0) {
                bool isBorder = false;

                for (int k = 0; k < 4; k++) {
                    int ni = i + di[k];
                    int nj = j + dj[k];

                    if (!isInside(labels, ni, nj) || labels(ni, nj) != label) {
                        isBorder = true;
                    }
                }

                if (isBorder) {
                    perim[label]++;
                }
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
        c.perimeter = perim[label];

        c.center = Point2f(
            (float)sumX[label] / area[label],
            (float)sumY[label] / area[label]
        );

        int width = maxX[label] - minX[label] + 1;
        int height = maxY[label] - minY[label] + 1;

        c.bbox = Rect(minX[label], minY[label], width, height);
        c.aspect = (float)width / height;

        if (c.perimeter > 0) {
            c.thinness = 4.0f * CV_PI * c.area / (c.perimeter * c.perimeter);
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

bool isStickerCandidate(ComponentInfo c, int imageArea)
{
    int minArea = max(20, (int)(0.00020f * imageArea));
    int maxArea = (int)(0.14f * imageArea);

    if (c.area < minArea) return false; //noise
    if (c.area > maxArea) return false; //background

    if (c.bbox.width < 5 || c.bbox.height < 5) return false; //lines or points

    if (c.aspect < 0.20f || c.aspect > 5.00f) return false;

    if (c.thinness < 0.06f) return false;

    if (isSkinLikeHSV(c.meanH, c.meanS, c.meanV)) return false; //hand

    return true;
}

vector<ComponentInfo> filterStickerCandidates(vector<ComponentInfo> components, Size imgSize)
{
    vector<ComponentInfo> candidates;
    int imageArea = imgSize.width * imgSize.height;

    for (ComponentInfo c : components) {
        if (isStickerCandidate(c, imageArea)) {
            candidates.push_back(c);
        }
    }

    sort(candidates.begin(), candidates.end(), [](ComponentInfo a, ComponentInfo b) {
        return a.area > b.area;
    });

    return candidates;
}

bool isFaceLikeComponent(ComponentInfo c, int imageArea)
{
    if (c.area < 0.0015f * imageArea) return false;
    if (c.area > 0.22f * imageArea) return false;

    if (c.bbox.width < 35 || c.bbox.height < 35) return false;

    if (c.aspect < 0.25f || c.aspect > 3.80f) return false;

    if (c.thinness < 0.06f) return false;

    if (isSkinLikeHSV(c.meanH, c.meanS, c.meanV)) return false;

    return true;
}

ComponentInfo findBestFaceCandidate(vector<ComponentInfo> components, Size imgSize)
{
    int imageArea = imgSize.width * imgSize.height;

    ComponentInfo best;
    float bestScore = -1.0f;

    for (ComponentInfo c : components) {
        if (!isFaceLikeComponent(c, imageArea)) {
            continue;
        }

        float areaScore = (float)c.area / imageArea;
        float aspectPenalty = fabs(1.0f - c.aspect);
        float compactnessBonus = c.thinness;

        float colorBonus = 0.0f;

        if (c.colorName == "BLUE" || c.colorName == "GREEN") {
            colorBonus = 0.45f;
        }
        else if (c.colorName == "YELLOW" || c.colorName == "RED") {
            colorBonus = 0.25f;
        }
        else if (c.colorName == "ORANGE") {
            colorBonus = 0.35f;
        }

        float score = 20.0f * areaScore + compactnessBonus + colorBonus - 0.15f * aspectPenalty;

        if (score > bestScore) {
            bestScore = score;
            best = c;
        }
    }

    return best;
}

bool isValidColoredFace(ComponentInfo face, Size imgSize)
{
    int imageArea = imgSize.width * imgSize.height;

    if (face.area <= 0) {
        return false;
    }

    if (face.colorName == "WHITE") {
        return false;
    }

    if (face.area < 0.006f * imageArea) { //small objects
        return false;
    }

    return true;
}

vector<ComponentInfo> keepLargestCloseCluster(vector<ComponentInfo> candidates)
{
    vector<ComponentInfo> bestCluster;
    int bestArea = 0;

    if (candidates.empty()) {
        return bestCluster;
    }

    for (int start = 0; start < candidates.size(); start++) {
        vector<ComponentInfo> cluster;
        cluster.push_back(candidates[start]);

        float baseDiag = componentDiag(candidates[start]);
        float maxDist = max(70.0f, baseDiag * 3.0f);

        bool changed = true;

        while (changed) {
            changed = false;

            for (ComponentInfo c : candidates) {
                bool alreadyInCluster = false;

                for (ComponentInfo existing : cluster) {
                    if (existing.label == c.label) {
                        alreadyInCluster = true;
                        break;
                    }
                }

                if (alreadyInCluster) {
                    continue;
                }

                for (ComponentInfo existing : cluster) {
                    float dist = pointDistance(c.center, existing.center);

                    if (dist < maxDist) {
                        cluster.push_back(c);
                        changed = true;
                        break;
                    }
                }
            }
        }

        int clusterArea = 0;

        for (ComponentInfo c : cluster) {
            clusterArea += c.area;
        }

        if (clusterArea > bestArea) {
            bestArea = clusterArea;
            bestCluster = cluster;
        }
    }

    sort(bestCluster.begin(), bestCluster.end(), [](ComponentInfo a, ComponentInfo b) {
        return a.area > b.area;
    });

    return bestCluster;
}

ComponentInfo buildFaceFromCluster(vector<ComponentInfo> cluster)
{
    ComponentInfo face;

    if (cluster.empty()) {
        return face;
    }

    int minX = 1000000000;
    int minY = 1000000000;
    int maxX = 0;
    int maxY = 0;

    long long sumArea = 0;
    long long sumX = 0;
    long long sumY = 0;
    long long sumH = 0;
    long long sumS = 0;
    long long sumV = 0;
    long long sumR = 0;
    long long sumG = 0;
    long long sumB = 0;

    for (ComponentInfo c : cluster) {
        minX = min(minX, c.bbox.x);
        minY = min(minY, c.bbox.y);
        maxX = max(maxX, c.bbox.x + c.bbox.width);
        maxY = max(maxY, c.bbox.y + c.bbox.height);

        sumArea += c.area;
        sumX += (long long)(c.center.x * c.area);
        sumY += (long long)(c.center.y * c.area);

        sumH += (long long)c.meanH * c.area;
        sumS += (long long)c.meanS * c.area;
        sumV += (long long)c.meanV * c.area;

        sumR += (long long)c.meanR * c.area;
        sumG += (long long)c.meanG * c.area;
        sumB += (long long)c.meanB * c.area;
    }

    face.label = -1;
    face.area = (int)sumArea;
    face.bbox = Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
    face.center = Point2f((float)sumX / sumArea, (float)sumY / sumArea);

    face.aspect = (float)face.bbox.width / face.bbox.height;
    face.perimeter = 2 * (face.bbox.width + face.bbox.height);

    if (face.perimeter > 0) {
        face.thinness = 4.0f * CV_PI * face.area / (face.perimeter * face.perimeter);
    }

    face.meanH = (int)(sumH / sumArea);
    face.meanS = (int)(sumS / sumArea);
    face.meanV = (int)(sumV / sumArea);

    face.meanR = (int)(sumR / sumArea);
    face.meanG = (int)(sumG / sumArea);
    face.meanB = (int)(sumB / sumArea);

    face.colorName = classifyRubikColor(
        face.meanH,
        face.meanS,
        face.meanV,
        face.meanR,
        face.meanG,
        face.meanB
    );

    return face;
}

vector<ComponentInfo> keepOnlyComponentsNearFace(vector<ComponentInfo> components, ComponentInfo face, Size imgSize) {
    vector<ComponentInfo> kept;

    if (face.area == 0) {
        return kept;
    }

    int margin = max(face.bbox.width, face.bbox.height) / 3;
    Rect faceZone = expandRectSafe(face.bbox, margin, imgSize);

    for (ComponentInfo c : components) {
        Point center((int)c.center.x, (int)c.center.y);

        if (faceZone.contains(center)) {
            kept.push_back(c);
        }
    }

    sort(kept.begin(), kept.end(), [](ComponentInfo a, ComponentInfo b) {
        return a.area > b.area;
    });

    return kept;
}

Rect refineFaceBoxByBestSquare(Mat_<uchar> mask, Rect initialBox, Size imgSize)
{
    initialBox = initialBox & Rect(0, 0, imgSize.width, imgSize.height);

    if (initialBox.width <= 0 || initialBox.height <= 0) {
        return initialBox;
    }

    Mat_<int> integral = buildObjectIntegralImage(mask);

    int maxSide = min(initialBox.width, initialBox.height);
    int minSide = max(35, (int)(0.45f * maxSide));

    Rect bestBox = initialBox;
    float bestScore = -1.0f;

    int sideStep = max(6, maxSide / 16);

    for (int side = maxSide; side >= minSide; side -= sideStep) {
        int step = max(4, side / 14);

        int yStart = initialBox.y;
        int yEnd = initialBox.y + initialBox.height - side;

        int xStart = initialBox.x;
        int xEnd = initialBox.x + initialBox.width - side;

        for (int y = yStart; y <= yEnd; y += step) {
            for (int x = xStart; x <= xEnd; x += step) {
                Rect candidate(x, y, side, side);

                int count = getObjectCountInRect(integral, candidate);
                float density = (float)count / (side * side);

                if (density < 0.25f) {
                    continue;
                }

                float score = density * 10000.0f + count * 0.03f - side * 0.25f;

                if (score > bestScore) {
                    bestScore = score;
                    bestBox = candidate;
                }
            }
        }
    }

    int margin = max(3, bestBox.width / 35);
    return expandRectSafe(bestBox, margin, imgSize);
}

Mat_<uchar> buildWhiteMask(Mat_<Vec3b> img, Mat_<uchar> H, Mat_<uchar> S, Mat_<uchar> V)
{
    Mat_<uchar> mask(img.rows, img.cols);
    mask.setTo(255);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            int B = img(i, j)[0];
            int G = img(i, j)[1];
            int R = img(i, j)[2];

            if (isWhiteRubikPixel(H(i, j), S(i, j), V(i, j), R, G, B)) {
                mask(i, j) = 0;
            }
        }
    }

    return mask;
}

int countColorPixelsInRect(Mat_<Vec3b> img, Mat_<uchar> H, Mat_<uchar> S, Mat_<uchar> V, Rect r)
{
    int count = 0;

    r = r & Rect(0, 0, img.cols, img.rows);

    for (int i = r.y; i < r.y + r.height; i++) {
        for (int j = r.x; j < r.x + r.width; j++) {
            int B = img(i, j)[0];
            int G = img(i, j)[1];
            int R = img(i, j)[2];

            if (isColoredRubikPixel(H(i, j), S(i, j), V(i, j), R, G, B)) {
                count++;
            }
        }
    }

    return count;
}

float averageSInRect(Mat_<uchar> S, Rect r)
{
    r = r & Rect(0, 0, S.cols, S.rows);

    if (r.width <= 0 || r.height <= 0) {
        return 255.0f;
    }

    long long sum = 0;
    int count = 0;

    for (int i = r.y; i < r.y + r.height; i++) {
        for (int j = r.x; j < r.x + r.width; j++) {
            sum += S(i, j);
            count++;
        }
    }

    if (count == 0) {
        return 255.0f;
    }

    return (float)sum / count;
}

float whiteDensityInRect(
    Mat_<Vec3b> img,
    Mat_<uchar> H,
    Mat_<uchar> S,
    Mat_<uchar> V,
    Rect r
) {
    r = r & Rect(0, 0, img.cols, img.rows);

    int whiteCount = 0;
    int totalCount = 0;

    for (int i = r.y; i < r.y + r.height; i++) {
        for (int j = r.x; j < r.x + r.width; j++) {
            int B = img(i, j)[0];
            int G = img(i, j)[1];
            int R = img(i, j)[2];

            if (isWhiteRubikPixel(H(i, j), S(i, j), V(i, j), R, G, B)) {
                whiteCount++;
            }

            totalCount++;
        }
    }

    if (totalCount == 0) {
        return 0.0f;
    }

    return (float)whiteCount / totalCount;
}

float skinDensityInRect(
    Mat_<Vec3b> img,
    Mat_<uchar> H,
    Mat_<uchar> S,
    Mat_<uchar> V,
    Rect r
) {
    r = r & Rect(0, 0, img.cols, img.rows);

    int skinCount = 0;
    int totalCount = 0;

    for (int i = r.y; i < r.y + r.height; i++) {
        for (int j = r.x; j < r.x + r.width; j++) {
            int B = img(i, j)[0];
            int G = img(i, j)[1];
            int R = img(i, j)[2];

            bool skin =
                H(i, j) >= 0 &&
                H(i, j) <= 35 &&
                S(i, j) >= 18 &&
                S(i, j) <= 145 &&
                V(i, j) >= 65 &&
                R > B + 8;

            if (skin) {
                skinCount++;
            }

            totalCount++;
        }
    }

    if (totalCount == 0) {
        return 0.0f;
    }

    return (float)skinCount / totalCount;
}

float saturationGridScoreInRect(Mat_<uchar> S, Rect box)
{
    box = box & Rect(0, 0, S.cols, S.rows);

    if (box.width < 60 || box.height < 60) {
        return 0.0f;
    }

    int band = max(2, box.width / 28);

    int x1 = box.x + box.width / 3;
    int x2 = box.x + 2 * box.width / 3;
    int y1 = box.y + box.height / 3;
    int y2 = box.y + 2 * box.height / 3;

    long long lineSum = 0;
    int lineCount = 0;

    for (int i = box.y; i < box.y + box.height; i++) {
        for (int dx = -band; dx <= band; dx++) {
            int j1 = x1 + dx;
            int j2 = x2 + dx;

            if (isInside(S, i, j1)) {
                lineSum += S(i, j1);
                lineCount++;
            }

            if (isInside(S, i, j2)) {
                lineSum += S(i, j2);
                lineCount++;
            }
        }
    }

    for (int j = box.x; j < box.x + box.width; j++) {
        for (int dy = -band; dy <= band; dy++) {
            int i1 = y1 + dy;
            int i2 = y2 + dy;

            if (isInside(S, i1, j)) {
                lineSum += S(i1, j);
                lineCount++;
            }

            if (isInside(S, i2, j)) {
                lineSum += S(i2, j);
                lineCount++;
            }
        }
    }

    if (lineCount == 0) {
        return 0.0f;
    }

    float lineMean = (float)lineSum / lineCount;

    Rect inner(
        box.x + box.width / 8,
        box.y + box.height / 8,
        box.width * 3 / 4,
        box.height * 3 / 4
    );

    float innerMean = averageSInRect(S, inner);

    float diff = lineMean - innerMean;

    if (diff < 0) {
        diff = 0;
    }

    return diff / 255.0f;
}

float darkGridScoreInRect(Mat_<uchar> V, Rect r)
{
    r = r & Rect(0, 0, V.cols, V.rows);

    if (r.width < 60 || r.height < 60) {
        return 0.0f;
    }

    double sum = 0.0;
    int total = 0;

    for (int i = r.y; i < r.y + r.height; i++) {
        for (int j = r.x; j < r.x + r.width; j++) {
            sum += V(i, j);
            total++;
        }
    }

    if (total == 0) {
        return 0.0f;
    }

    int threshold = (int)(sum / total - 12.0);
    threshold = max(45, min(190, threshold));

    int band = max(2, r.width / 28);

    int x1 = r.x + r.width / 3;
    int x2 = r.x + 2 * r.width / 3;
    int y1 = r.y + r.height / 3;
    int y2 = r.y + 2 * r.height / 3;

    int darkPixels = 0;
    int checkedPixels = 0;

    for (int i = r.y; i < r.y + r.height; i++) {
        for (int dx = -band; dx <= band; dx++) {
            int j1 = x1 + dx;
            int j2 = x2 + dx;

            if (isInside(V, i, j1)) {
                checkedPixels++;
                if (V(i, j1) < threshold) darkPixels++;
            }

            if (isInside(V, i, j2)) {
                checkedPixels++;
                if (V(i, j2) < threshold) darkPixels++;
            }
        }
    }

    for (int j = r.x; j < r.x + r.width; j++) {
        for (int dy = -band; dy <= band; dy++) {
            int i1 = y1 + dy;
            int i2 = y2 + dy;

            if (isInside(V, i1, j)) {
                checkedPixels++;
                if (V(i1, j) < threshold) darkPixels++;
            }

            if (isInside(V, i2, j)) {
                checkedPixels++;
                if (V(i2, j) < threshold) darkPixels++;
            }
        }
    }

    if (checkedPixels == 0) {
        return 0.0f;
    }

    return (float)darkPixels / checkedPixels;
}

float whiteNineCellScore(
    Mat_<Vec3b> img,
    Mat_<uchar> H,
    Mat_<uchar> S,
    Mat_<uchar> V,
    Rect box
) {
    box = box & Rect(0, 0, img.cols, img.rows);

    if (box.width < 60 || box.height < 60) {
        return 0.0f;
    }

    int goodCells = 0;
    float totalDensity = 0.0f;
    float totalAvgS = 0.0f;
    float minDensity = 1.0f;
    float maxDensity = 0.0f;

    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            int x1 = box.x + col * box.width / 3;
            int x2 = box.x + (col + 1) * box.width / 3;
            int y1 = box.y + row * box.height / 3;
            int y2 = box.y + (row + 1) * box.height / 3;

            int marginX = max(3, (x2 - x1) / 5);
            int marginY = max(3, (y2 - y1) / 5);

            Rect cell(
                x1 + marginX,
                y1 + marginY,
                max(1, x2 - x1 - 2 * marginX),
                max(1, y2 - y1 - 2 * marginY)
            );

            float density = whiteDensityInRect(img, H, S, V, cell);
            float avgS = averageSInRect(S, cell);

            totalDensity += density;
            totalAvgS += avgS;

            minDensity = min(minDensity, density);
            maxDensity = max(maxDensity, density);

            if (density >= 0.24f && avgS <= 78.0f) {
                goodCells++;
            }
        }
    }

    // Important:
    // For the white face, we want almost all 9 cells.
    // This rejects the false box above the cube.
    if (goodCells < 8) {
        return 0.0f;
    }

    if (minDensity < 0.08f) {
        return 0.0f;
    }

    float avgDensity = totalDensity / 9.0f;
    float avgS = totalAvgS / 9.0f;

    float lowSBonus = max(0.0f, (85.0f - avgS) / 85.0f);
    float balancePenalty = maxDensity - minDensity;

    return
        avgDensity +
        0.10f * goodCells +
        lowSBonus -
        0.35f * balancePenalty;
}

float strictWhiteGridScoreInRect(Mat_<uchar> S, Mat_<uchar> V, Rect box)
{
    box = box & Rect(0, 0, S.cols, S.rows);

    if (box.width < 60 || box.height < 60) {
        return 0.0f;
    }

    double sumV = 0.0;
    int total = 0;

    for (int i = box.y; i < box.y + box.height; i++) {
        for (int j = box.x; j < box.x + box.width; j++) {
            sumV += V(i, j);
            total++;
        }
    }

    if (total == 0) {
        return 0.0f;
    }

    int thresholdV = (int)(sumV / total - 12.0);
    thresholdV = max(45, min(190, thresholdV));

    Rect inner(
        box.x + box.width / 8,
        box.y + box.height / 8,
        box.width * 3 / 4,
        box.height * 3 / 4
    );

    float innerMeanS = averageSInRect(S, inner);

    int band = max(2, box.width / 30);

    int xLines[2] = {
        box.x + box.width / 3,
        box.x + 2 * box.width / 3
    };

    int yLines[2] = {
        box.y + box.height / 3,
        box.y + 2 * box.height / 3
    };

    float scores[4];
    int scoreIndex = 0;

    for (int lineIndex = 0; lineIndex < 2; lineIndex++) {
        int x = xLines[lineIndex];

        int darkCount = 0;
        int count = 0;
        long long sumSLine = 0;

        for (int i = box.y; i < box.y + box.height; i++) {
            for (int dx = -band; dx <= band; dx++) {
                int j = x + dx;

                if (!isInside(V, i, j)) {
                    continue;
                }

                if (V(i, j) < thresholdV) {
                    darkCount++;
                }

                sumSLine += S(i, j);
                count++;
            }
        }

        if (count == 0) {
            scores[scoreIndex++] = 0.0f;
            continue;
        }

        float darkRatio = (float)darkCount / count;
        float meanSLine = (float)sumSLine / count;
        float sContrast = max(0.0f, (meanSLine - innerMeanS) / 255.0f);

        scores[scoreIndex++] = darkRatio + 2.0f * sContrast;
    }

    for (int lineIndex = 0; lineIndex < 2; lineIndex++) {
        int y = yLines[lineIndex];

        int darkCount = 0;
        int count = 0;
        long long sumSLine = 0;

        for (int j = box.x; j < box.x + box.width; j++) {
            for (int dy = -band; dy <= band; dy++) {
                int i = y + dy;

                if (!isInside(V, i, j)) {
                    continue;
                }

                if (V(i, j) < thresholdV) {
                    darkCount++;
                }

                sumSLine += S(i, j);
                count++;
            }
        }

        if (count == 0) {
            scores[scoreIndex++] = 0.0f;
            continue;
        }

        float darkRatio = (float)darkCount / count;
        float meanSLine = (float)sumSLine / count;
        float sContrast = max(0.0f, (meanSLine - innerMeanS) / 255.0f);

        scores[scoreIndex++] = darkRatio + 2.0f * sContrast;
    }

    float minScore = scores[0];

    for (int i = 1; i < 4; i++) {
        minScore = min(minScore, scores[i]);
    }

    return minScore;
}
ComponentInfo findBestWhiteFaceCandidate(
    Mat_<Vec3b> img,
    Mat_<uchar> H,
    Mat_<uchar> S,
    Mat_<uchar> V,
    Mat_<uchar> coloredMask
) {
    ComponentInfo face;

    int minDim = min(img.rows, img.cols);

    int minSide = max(220, minDim / 5);
    int maxSide = max(minSide + 1, (int)(0.58f * minDim));

    Rect bestBox;
    float bestScore = -1.0f;
    int bestWhiteCount = 0;

    Mat_<uchar> whiteMask = buildWhiteMask(img, H, S, V);
    Mat_<int> whiteIntegral = buildObjectIntegralImage(whiteMask);

    int sideStep = max(12, maxSide / 24);

    int xStartGlobal = (int)(img.cols * 0.34f);
    int xEndGlobal = img.cols - 20;

    for (int side = maxSide; side >= minSide; side -= sideStep) {
        int step = max(7, side / 18);

        for (int y = 20; y + side < img.rows - 20; y += step) {
            for (int x = xStartGlobal; x + side < xEndGlobal; x += step) {
                Rect candidate(x, y, side, side);

                Point2f center(
                    candidate.x + candidate.width / 2.0f,
                    candidate.y + candidate.height / 2.0f
                );

                // Important:
                // The false detection was too high.
                // The cube face in your images is around middle / lower-middle.
                if (center.y < img.rows * 0.24f) {
                    continue;
                }

                int whiteCount = getObjectCountInRect(whiteIntegral, candidate);
                float whiteDensity = (float)whiteCount / (side * side);

                if (whiteDensity < 0.22f || whiteDensity > 0.94f) {
                    continue;
                }

                float avgS = averageSInRect(S, candidate);

                if (avgS > 82.0f) {
                    continue;
                }

                float skinDensity = skinDensityInRect(img, H, S, V, candidate);

                if (skinDensity > 0.24f) {
                    continue;
                }

                float cellScore = whiteNineCellScore(img, H, S, V, candidate);

                if (cellScore < 0.60f) {
                    continue;
                }

                float vGridScore = darkGridScoreInRect(V, candidate);
                float sGridScore = saturationGridScoreInRect(S, candidate);
                float strictGridScore = strictWhiteGridScoreInRect(S, V, candidate);

                // This is the main fix:
                // all 4 internal grid lines must be present enough.
                if (strictGridScore < 0.006f) {
                    continue;
                }

                float centerPreference = center.x / img.cols;
                float yPreference = 1.0f - fabs((center.y / img.rows) - 0.45f);

                if (yPreference < 0.0f) {
                    yPreference = 0.0f;
                }

                float lowSScore = max(0.0f, (85.0f - avgS) / 85.0f);
                float sizeScore = (float)side / (float)minDim;

                float score =
                    whiteDensity * 6500.0f +
                    cellScore * 16000.0f +
                    vGridScore * 35000.0f +
                    sGridScore * 80000.0f +
                    strictGridScore * 150000.0f +
                    lowSScore * 5000.0f +
                    sizeScore * 7000.0f +
                    centerPreference * 900.0f +
                    yPreference * 5000.0f -
                    skinDensity * 22000.0f;

                if (score > bestScore) {
                    bestScore = score;
                    bestBox = candidate;
                    bestWhiteCount = whiteCount;
                }
            }
        }
    }

    if (bestScore < 0.0f) {
        return face;
    }

    int margin = max(3, bestBox.width / 45);
    bestBox = expandRectSafe(bestBox, margin, img.size());

    long long sumH = 0;
    long long sumS = 0;
    long long sumV = 0;
    long long sumR = 0;
    long long sumG = 0;
    long long sumB = 0;
    int count = 0;

    for (int i = bestBox.y; i < bestBox.y + bestBox.height; i++) {
        for (int j = bestBox.x; j < bestBox.x + bestBox.width; j++) {
            int B = img(i, j)[0];
            int G = img(i, j)[1];
            int R = img(i, j)[2];

            if (isWhiteRubikPixel(H(i, j), S(i, j), V(i, j), R, G, B)) {
                sumH += H(i, j);
                sumS += S(i, j);
                sumV += V(i, j);
                sumR += R;
                sumG += G;
                sumB += B;
                count++;
            }
        }
    }

    if (count == 0) {
        return face;
    }

    face.label = -2;
    face.area = bestWhiteCount;
    face.bbox = bestBox;

    face.center = Point2f(
        bestBox.x + bestBox.width / 2.0f,
        bestBox.y + bestBox.height / 2.0f
    );

    face.aspect = 1.0f;
    face.perimeter = 2 * (bestBox.width + bestBox.height);

    if (face.perimeter > 0) {
        face.thinness = 4.0f * CV_PI * face.area / (face.perimeter * face.perimeter);
    }

    face.meanH = (int)(sumH / count);
    face.meanS = (int)(sumS / count);
    face.meanV = (int)(sumV / count);
    face.meanR = (int)(sumR / count);
    face.meanG = (int)(sumG / count);
    face.meanB = (int)(sumB / count);

    face.colorName = "WHITE";

    return face;
}

Mat_<Vec3b> drawCandidates(Mat_<Vec3b> img, vector<ComponentInfo> candidates)
{
    Mat_<Vec3b> result = img.clone();

    int count = 0;

    for (ComponentInfo c : candidates) {
        Vec3b boxColor = Vec3b(0, 255, 0);

        if (count < 9) {
            boxColor = Vec3b(0, 0, 255);
        }

        rectangle(result, c.bbox, boxColor, 2);
        circle(result, Point((int)c.center.x, (int)c.center.y), 3, Vec3b(255, 0, 0), FILLED);

        string text = c.colorName + " A=" + to_string(c.area);

        putText(
            result,
            text,
            Point(c.bbox.x, max(0, c.bbox.y - 5)),
            FONT_HERSHEY_SIMPLEX,
            0.45,
            Scalar(0, 0, 255),
            1
        );

        count++;
    }

    return result;
}

Mat_<Vec3b> drawFaceAndSimpleGrid(Mat_<Vec3b> img, ComponentInfo face, Mat_<uchar> cleanMask)
{
    Mat_<Vec3b> result = img.clone();

    if (face.area == 0) {
        putText(
            result,
            "No face candidate found",
            Point(20, 30),
            FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar(0, 0, 255),
            2
        );

        return result;
    }

    Rect box;

    if (face.label == -2) {
        box = face.bbox;
    }
    else {
        box = refineFaceBoxByBestSquare(cleanMask, face.bbox, img.size());
    }

    rectangle(result, box, Vec3b(0, 0, 255), 3);

    for (int k = 1; k <= 2; k++) {
        int x = box.x + k * box.width / 3;
        int y = box.y + k * box.height / 3;

        line(result, Point(x, box.y), Point(x, box.y + box.height), Scalar(255, 0, 0), 2);
        line(result, Point(box.x, y), Point(box.x + box.width, y), Scalar(255, 0, 0), 2);
    }

    string text = "FACE " + face.colorName + " A=" + to_string(face.area);

    putText(
        result,
        text,
        Point(box.x, max(0, box.y - 8)),
        FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar(0, 0, 255),
        2
    );

    return result;
}

void printCandidates(string title, vector<ComponentInfo> candidates)
{
    cout << "\n" << title << ": " << candidates.size() << endl;

    for (int i = 0; i < candidates.size(); i++) {
        ComponentInfo c = candidates[i];

        cout << i + 1
             << " label=" << c.label
             << " area=" << c.area
             << " center=(" << c.center.x << "," << c.center.y << ")"
             << " bbox=" << c.bbox.width << "x" << c.bbox.height
             << " aspect=" << c.aspect
             << " thinness=" << c.thinness
             << " HSV=(" << c.meanH << "," << c.meanS << "," << c.meanV << ")"
             << " RGB=(" << c.meanR << "," << c.meanG << "," << c.meanB << ")"
             << " color=" << c.colorName
             << endl;
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
         << " thinness=" << face.thinness
         << " HSV=(" << face.meanH << "," << face.meanS << "," << face.meanV << ")"
         << " RGB=(" << face.meanR << "," << face.meanG << "," << face.meanB << ")"
         << " color=" << face.colorName
         << endl;
}

void project()
{
    string filename;
    filename = "Project/portocaliu.bmp";
    Mat_<Vec3b> img = imread(filename);

    Mat_<uchar> H, S, V;

    Mat_<uchar> coloredMask = buildRubikMask(img, H, S, V);

    Mat_<uchar> strel = squareStrel3();
    Mat_<uchar> opened = openingOp(coloredMask, strel);
    Mat_<uchar> cleanMask = opened;

    Mat_<int> labels(cleanMask.rows, cleanMask.cols, 0);
    bfs_connected_components(cleanMask, labels, true);

    vector<ComponentInfo> components = extractComponentsFromLabels(labels, H, S, V, img);
    vector<ComponentInfo> stickerCandidates = filterStickerCandidates(components, img.size());

    ComponentInfo coloredFace = findBestFaceCandidate(components, img.size());

    ComponentInfo face;

    if (isValidColoredFace(coloredFace, img.size())) {
        face = coloredFace;
        cout << "Valid colored face found: " << face.colorName << endl;
    }
    else {
        cout << "No valid colored face found. Trying white detection" << endl;
        // TODO
    }

    vector<ComponentInfo> finalCandidates;

    if (face.area > 0 && face.label != -2) {
        finalCandidates = keepOnlyComponentsNearFace(components, face, img.size());
    }
    else {
        finalCandidates.clear();
    }

    Mat_<Vec3b> allCandidatesImage = drawCandidates(img, stickerCandidates);
    Mat_<Vec3b> finalCandidatesImage = drawCandidates(img, finalCandidates);
    Mat_<Vec3b> faceGridImage = drawFaceAndSimpleGrid(img, face, cleanMask);

    printCandidates("All sticker-like candidates", stickerCandidates);
    printFace(face);
    printCandidates("Final close candidates near cube face", finalCandidates);

    imshow("1 Original", img);
    imshow("2 H channel", H);
    imshow("3 S channel", S);
    imshow("4 V channel", V);
    imshow("5 Colored mask", coloredMask);
    imshow("6 Clean mask", cleanMask);
    displayLabels(labels, "8 Connected components");
    // imshow("8 All sticker-like candidates", allCandidatesImage);
    // imshow("9 Final close cube candidates", finalCandidatesImage);
    imshow("10 Detected face and simple 3x3 grid", faceGridImage);

    waitKey(0);
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
