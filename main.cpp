#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;
using namespace cv;

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

void bfs_connected_components(Mat_<uchar> img, Mat_<int> labels, bool use8Neighbors) {
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

void twopass_connected_components(Mat_<uchar> img, Mat_<int>& labels, bool use8Neighbors) {
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

    displayLabels(labels, "between passes");

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

void project() {
    VideoCapture cap(0, CAP_AVFOUNDATION);
    if (!cap.isOpened()) {
        cap.open(0);
    }
    if (!cap.isOpened()) {
        cout << "Could not open the laptop camera. Check macOS Camera permissions for CLion/Terminal." << endl;
        return;
    }
    cout << "Camera opened. Press ESC or q to close." << endl;
    Mat_<Vec3b> frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "Received empty frame from camera. Stopping." << endl;
            break;
        }
        rectangle(frame, Point(10, 10), Point(370, 60), Scalar(0, 0, 0), FILLED);
        putText(frame, "Lab Project", Point(20, 35),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
        putText(frame, "Press q or ESC to stop live", Point(20, 55),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        imshow("Laptop Camera", frame);
        int key = waitKey(30);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
    }
    cap.release();
    destroyWindow("Laptop Camera");
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
            case 15:
                project();
                break;
        }
    }
    while (op!=0);
    return 0;
}
