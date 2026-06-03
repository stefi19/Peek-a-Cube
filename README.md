# Peek-a-Cube: Rubik's Cube Face Detection

Peek-a-Cube is an image processing project that detects the front face of a Rubik's cube, extracts the 9 visible stickers from that face, and prints the detected 3x3 color matrix.

The project code is part of the same `main.cpp` file as the laboratory exercises. The Rubik project section starts at:

```cpp
// PROJECT
```

The main entry point for the Rubik detection pipeline is:

```cpp
Mat_<Vec3b> processRubikFrame(Mat_<Vec3b> inputImg, bool showDebugWindows, bool printInfo)
```

This function receives a BGR image, detects the Rubik face region first, and then searches for stickers only inside that detected region. The algorithm does not search for a 3x3 grid over the whole image, because doing so can accidentally include the hand, the background, or stickers from a side face of the cube.

---

## Table of Contents

- [Project Goal](#project-goal)
- [What the Application Does](#what-the-application-does)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Build and Run](#build-and-run)
- [Application Menu](#application-menu)
- [Main Algorithm Idea](#main-algorithm-idea)
- [Complete Pipeline](#complete-pipeline)
- [HSV and Color Explanation](#hsv-and-color-explanation)
- [Rubik Face Detection](#rubik-face-detection)
- [White Face Detection](#white-face-detection)
- [Sticker Extraction Inside the ROI](#sticker-extraction-inside-the-roi)
- [Adaptive Edge Detection Inside the ROI](#adaptive-edge-detection-inside-the-roi)
- [Sticker Color Classification](#sticker-color-classification)
- [Debug Windows](#debug-windows)
- [Important Functions](#important-functions)
- [Limitations](#limitations)
- [How to Test](#how-to-test)
- [Common Problems and Fixes](#common-problems-and-fixes)
- [Code Style Notes](#code-style-notes)
- [Summary](#summary)

---

## Project Goal

The goal of the project is to automatically detect one visible face of a Rubik's cube from either a static image or a live camera frame.

The expected output is:

1. a tight outline around the detected front face;
2. exactly 9 sticker cells drawn inside that detected face;
3. a 3x3 color matrix printed in the console;
4. debug windows that show the important processing steps.

This is not just a problem of finding colored squares. Real images contain several complications:

- the hand holding the cube can have colors close to red or orange;
- a side face of the cube can also be visible;
- lighting can change saturation and brightness;
- the white face is hard to detect by color alone because it has low saturation;
- black separators between stickers can break or merge components;
- reflections or the logo on a white sticker can confuse classification.

Because of these issues, the algorithm follows one important rule:

First detect the main cube face, then process only that face region.

This is the most important design choice in the current implementation.

---

## What the Application Does

The application can run in two modes:

1. static image mode, using images from the `Project/` folder;
2. live camera mode, using a webcam.

For every image or frame, the application:

1. resizes the image if it is too large;
2. converts the image to HSV using the custom `convertRGBtoHSV` function;
3. builds a mask for colored Rubik pixels;
4. cleans the mask using morphological operations;
5. detects connected components;
6. chooses the most likely Rubik face region;
7. if no colored face is found, tries a Canny-based fallback for the white face;
8. warps the detected face region into a 300x300 image;
9. searches for stickers only inside that warped ROI;
10. reconstructs a 3x3 grid inside the ROI;
11. classifies each sticker using the center of its cell;
12. draws the result and prints the color matrix.

---

## Project Structure

The relevant project structure is:

```text
Peek-a-Cube/
├── main.cpp
├── CMakeLists.txt
├── README.md
├── Project/
│   ├── portocaliu.bmp
│   ├── rosu.bmp
│   ├── galben.bmp
│   ├── verde.bmp
│   ├── albastru.bmp
│   ├── alb.bmp
│   ├── original_resized.png
│   ├── H channel.png
│   ├── S channel.png
│   ├── V channel.png
│   ├── colored mask.png
│   ├── clean colored mask.png
│   ├── connected components.png
│   ├── final.png
│   └── prezentare.html
└── build/
    └── Lab1
```

Important files:

- `main.cpp`: contains all laboratory functions and the Rubik project code;
- `CMakeLists.txt`: configures the project build and links OpenCV;
- `Project/*.bmp`: test images for the main cube face colors;
- `Project/prezentare.html`: visual project presentation;
- `README.md`: this documentation file.

The CMake target is named:

```text
Lab1
```

So the executable is run with:

```bash
./build/Lab1
```

---

## Dependencies

The project uses C++ and OpenCV.

Required:

- a C++ compiler with C++11 support;
- CMake;
- OpenCV;
- a system that can display OpenCV windows through `imshow`;
- a webcam, only for live mode.

On macOS with Homebrew:

```bash
brew install cmake opencv
```

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install cmake libopencv-dev
```

In `CMakeLists.txt`, the macOS OpenCV path is currently set as:

```cmake
set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
```

If OpenCV is installed somewhere else, this path may need to be changed.

---

## Build and Run

From the project root:

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

If the `build` directory already exists, this is enough:

```bash
cmake --build build
```

Run the executable:

```bash
./build/Lab1
```

If the build succeeds, the output should contain:

```text
[100%] Built target Lab1
```

---

## Application Menu

At startup, the main laboratory menu appears:

```text
Menu:
 1 - Lab1
 2 - Lab2
 3 - Lab3
 ...
 15 - Project
 0 - Exit

Option:
```

For the Rubik project, choose:

```text
15
```

Then the project menu appears:

```text
PROJECT MENU:
1 - Run on images from Project/ folder
2 - Open camera and detect live
0 - Exit project
Option:
```

Option `1` runs the detector on the six test images:

```text
Project/portocaliu.bmp
Project/rosu.bmp
Project/galben.bmp
Project/verde.bmp
Project/albastru.bmp
Project/alb.bmp
```

Option `2` opens the camera and processes live frames.

The menu behavior is intentionally unchanged. The project remains integrated into option `15`.

---

## Main Algorithm Idea

The central idea is:

Do not search for the 3x3 sticker grid in the whole image.

The algorithm first detects the main Rubik face. Only after the face is found does it crop and warp that face into a square ROI. Sticker detection and sticker classification are then performed only inside that ROI.

This matters because an input image can contain:

- stickers from a side face;
- hand pixels;
- colored background pixels;
- reflections;
- black cube edges;
- square objects in the background.

If the 3x3 grid were searched globally, the algorithm could mix elements from different image regions. For example, it could take 6 stickers from the front face and 3 stickers from a side face. It could also mistake part of the hand or background for a sticker.

The current pipeline avoids that by making the face ROI the boundary for all later sticker work.

---

## Complete Pipeline

The current processing pipeline is:

```text
BGR image
   |
   v
Resize image if needed
   |
   v
Convert BGR -> HSV with convertRGBtoHSV
   |
   v
Build colored Rubik mask
   |
   v
Opening 3x3 + closing 5x5
   |
   v
BFS connected components
   |
   v
Extract component properties
   |
   v
Choose the best colored face candidate
   |
   +-- if no colored face is found:
   |       use Canny-based white face detection
   |
   v
Rubik face ROI
   |
   v
Warp ROI to 300x300
   |
   v
HSV + adaptive edge detection inside ROI
   |
   v
Sticker contours inside ROI
   |
   v
Fallback to 3x3 split if not all stickers are found
   |
   v
Classify color from the center of each cell
   |
   v
Draw face + 9 sticker cells + print matrix
```

The order is important. Sticker detection depends on the ROI. It is not a global search over the image.

---

## HSV and Color Explanation

The original image is BGR because OpenCV stores color images as BGR by default. For classification, the project uses HSV because it separates:

- hue, which represents the color type;
- saturation, which represents color intensity/purity;
- value, which represents brightness.

The conversion function used by the project is:

```cpp
Mat_<Vec3b> convertRGBtoHSV(Mat_<Vec3b> img)
```

This is a custom conversion, not OpenCV's `cvtColor`. The resulting channels are:

- `H` in the range 0..255;
- `S` in the range 0..255;
- `V` in the range 0..255.

In the usual HSV model, hue is an angle from 0 to 360 degrees. In this project it is normalized:

```text
H_normalized = H_degrees * 255 / 360
```

Approximate ranges:

```text
red       H close to 0 or close to 255
orange    H roughly in 7..45
yellow    H roughly in 28..82
green     H roughly in 65..155
blue      H roughly in 110..235
white     low S, high V
```

For white, hue is not very useful. White is mainly described by low saturation and high brightness.

---

## Rubik Face Detection

Face detection starts from colored Rubik pixels, not from the sticker grid.

The function:

```cpp
Mat_<uchar> buildColoredRubikMask(Mat_<Vec3b> img, Mat_<uchar>& H, Mat_<uchar>& S, Mat_<uchar>& V)
```

does the following:

1. converts the BGR image to HSV;
2. splits the HSV image into `H`, `S`, and `V`;
3. visits every pixel;
4. checks if the pixel matches one of the colored Rubik stickers;
5. excludes skin-colored pixels;
6. builds a binary mask.

The mask convention is:

```text
0   = object pixel, meaning colored Rubik pixel
255 = background
```

After building the mask, the code applies:

```cpp
openingOp(coloredMask, strel3)
closingOp(cleanColoredMask, strel5)
```

Their roles are:

- opening 3x3 removes small noise;
- closing 5x5 connects nearby areas and reduces gaps between stickers.

Then connected components are computed with BFS:

```cpp
bfs_connected_components(cleanColoredMask, coloredLabels, true)
```

For every connected component, the code computes:

- area;
- center;
- bounding box;
- aspect ratio;
- density inside the bounding box;
- mean HSV values;
- mean RGB values;
- estimated color name.

The relevant functions are:

```cpp
extractComponentsFromLabels(...)
isPossibleCubeFace(...)
findBestComponentFace(...)
```

A component can be a possible Rubik face if:

- it is not too small;
- it is not too large;
- its width and height are large enough;
- its aspect ratio is acceptable;
- its bounding-box density is high enough;
- its color is not white and not unknown.

White is not handled through the colored component detector. It has its own Canny fallback.

The best colored component is selected with:

```text
score = 20.0 * areaScore
      + 2.0 * densityScore
      - 0.5 * aspectPenalty
```

Where:

- `areaScore` favors larger components;
- `densityScore` favors compact components;
- `aspectPenalty` penalizes shapes that are too far from a square/rectangle face.

---

## White Face Detection

The white face is treated separately because it is not strongly saturated. White stickers usually have:

- low saturation;
- high value/brightness;
- visible black grid lines between stickers.

If no colored face is found, the code calls:

```cpp
ComponentInfo findWhiteFaceWithCanny(Mat_<Vec3b> img, Mat_<uchar> H, Mat_<uchar> S, Mat_<uchar> V, bool showCanny)
```

Main steps:

1. blur the `V` channel;
2. apply Canny on `V`;
3. scan square candidate windows;
4. for each candidate window, check:
   - average saturation is low enough;
   - average brightness is high enough;
   - white pixel density is high enough;
   - expected inner grid lines have enough edges;
   - outer borders have enough edges.

For a white face, the `V` channel is useful because black separators contrast strongly with bright white stickers.

Helper functions:

```cpp
averageValueInRect(...)
whiteDensityInRectForCanny(...)
cannyGridScore(...)
outerEdgeScore(...)
```

`cannyGridScore` checks for edges near the expected 1/3 and 2/3 grid lines.

`outerEdgeScore` checks whether the outside border of the face is visible.

---

## Sticker Extraction Inside the ROI

After the face is found, the detected component is converted into a simple face structure:

```cpp
struct RubikFace {
    vector<Point2f> corners;
    Rect bbox;
    bool valid = false;
};
```

The function:

```cpp
RubikFace faceFromComponent(ComponentInfo component, Size imageSize)
```

builds the four face corners from the detected bounding box.

Then:

```cpp
extractRubikStickers(...)
```

does the sticker processing:

1. creates a destination square of 300x300;
2. computes the perspective transform with `getPerspectiveTransform`;
3. applies `warpPerspective`;
4. computes HSV inside the warped ROI;
5. builds adaptive edges inside the ROI;
6. searches for sticker contours only inside the ROI;
7. fills the 9 final cells;
8. classifies each cell;
9. maps stickers back to the original image.

The 300x300 size is chosen because it makes the grid simple:

```text
300 / 3 = 100
```

So each ideal cell is 100x100 pixels.

For each cell:

```text
row = 0..2
col = 0..2
cellBox = Rect(col * 100, row * 100, 100, 100)
```

Important details:

- the grid never leaves the ROI;
- sticker contours are searched inside the ROI, not in the full image;
- if good contours are not found for all 9 cells, the code falls back to splitting the warped ROI into a 3x3 grid;
- the fallback still stays inside the face ROI, so it does not include hand or background.

---

## Adaptive Edge Detection Inside the ROI

Inside the ROI, the code uses:

```cpp
buildAdaptiveRubikEdges(...)
```

This function receives the `S` and `V` channels and produces:

- `edgeMap`;
- `closedMap`.

For low-saturation faces, usually white faces:

```text
use the V channel
```

For colored faces:

```text
combine S + V + grayscale
```

Reasoning:

- on white faces, the useful contrast is brightness contrast between white stickers and black grid lines;
- on colored faces, saturation and brightness help separate stickers;
- grayscale can help capture black separator edges.

After Canny, the code applies:

```cpp
dilate(...)
morphologyEx(..., MORPH_CLOSE, ...)
```

These operations connect broken edges and make sticker contours more stable.

---

## Sticker Color Classification

Sticker classification does not use the full cell. It uses only the central area:

```cpp
Rect sample(
    cellBox.x + cellBox.width * 0.27,
    cellBox.y + cellBox.height * 0.27,
    cellBox.width * 0.46,
    cellBox.height * 0.46
);
```

This means roughly the center of the cell is sampled, while avoiding:

- sticker borders;
- black grid lines;
- mixed pixels near neighboring stickers;
- strong edge pixels;
- small reflections near borders;
- logo pixels if they do not dominate the center.

The main functions are:

```cpp
classifyStickerPixel(...)
classifyStickerCentralRegion(...)
```

For each pixel in the central region:

1. ignore pixels that are on detected edges;
2. ignore pixels that are too dark;
3. ignore very bright, low-saturation pixels that are likely highlights;
4. vote for the pixel color;
5. store H, S, V values for median calculation;
6. choose the dominant color.

White is accepted only if:

- enough pixels vote for white;
- median saturation is low;
- median value is high.

This prevents a small reflection or logo from turning a sticker into white.

For red and orange, RGB dominance is also checked:

- red requires strong `R` dominance;
- orange requires dominant `R`, but with enough `G`;
- orange is separated from yellow using the `G/R` ratio;
- red is separated from orange by stronger red dominance.

---

## Debug Windows

When `showDebugWindows` is `true`, the application opens useful debug windows.

Main windows:

```text
1 Original resized
2 H channel
3 S channel
4 V channel
5 Colored mask
6 Clean colored mask
7 Colored connected components
8 ROI warped face
9 ROI adaptive edges
10 ROI closed edges
11 ROI sticker candidates
12 Final Rubik grid
White Canny on V channel
```

Some windows appear only in certain cases:

- `White Canny on V channel` appears when the white-face fallback is used;
- colored mask windows appear when colored face detection is used;
- ROI windows appear only if a face is found.

How to read them:

- `H channel`: hue distribution;
- `S channel`: color saturation;
- `V channel`: brightness;
- `Colored mask`: pixels considered colored Rubik stickers;
- `Clean colored mask`: mask after morphological cleanup;
- `ROI warped face`: the detected face warped to 300x300;
- `ROI adaptive edges`: edges detected inside the ROI;
- `ROI closed edges`: edges after dilation and closing;
- `ROI sticker candidates`: detected sticker-like contours inside the ROI;
- `Final Rubik grid`: final output with face outline and 9 cells.

---

## Important Functions

### `processRubikFrame`

```cpp
Mat_<Vec3b> processRubikFrame(Mat_<Vec3b> inputImg, bool showDebugWindows, bool printInfo)
```

This is the main function of the Rubik project.

It:

- receives the input image;
- detects the face ROI;
- extracts stickers inside the ROI;
- classifies sticker colors;
- draws the final result;
- shows debug windows if requested.

This function is used both by static image mode and live camera mode.

### `detectRubikFaceROI`

```cpp
ComponentInfo detectRubikFaceROI(...)
```

Detects the Rubik face using:

- color mask;
- morphology;
- BFS connected components;
- geometric scoring;
- Canny fallback for white faces.

### `extractRubikStickers`

```cpp
vector<StickerInfo> extractRubikStickers(...)
```

Receives an already detected face and works only inside that face area.

It returns the final stickers used for drawing and for the color matrix.

### `classifyStickerCentralRegion`

```cpp
string classifyStickerCentralRegion(...)
```

Classifies one sticker using the center of its cell.

This is one of the most important functions for stable color detection.

### `drawFaceAndStickerGrid`

```cpp
Mat_<Vec3b> drawFaceAndStickerGrid(...)
```

Draws:

- the detected face outline;
- the 9 sticker cells;
- row/column/color labels;
- the 3x3 matrix in the console if `printMatrix` is `true`.

### `tryOpenCamera`

```cpp
bool tryOpenCamera(VideoCapture& cap)
```

Tries multiple camera indices and backends until a valid frame is received.

### `project`

```cpp
void project()
```

Contains the project menu:

```text
1 - Run on images from Project/ folder
2 - Open camera and detect live
0 - Exit project
```

The menu behavior has not been changed.

---

## Limitations

The algorithm is designed for reasonably clear images where the front face is visible.

Problems can appear when:

- the cube is very far from the camera;
- the hand covers too much of the front face;
- the image is very dark;
- reflections are very strong;
- the cube is rotated so much that the front face becomes very narrow;
- the background has colors similar to the stickers;
- the camera changes white balance strongly;
- a side face becomes larger or more visible than the intended front face.

The ROI-first pipeline reduces the risk of mixing the front face with side faces or background, but it still depends on the face ROI being detected correctly.

---

## How to Test

### Build test

From the project root:

```bash
cmake --build build
```

If there is no `build` folder:

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

### Static image test

Run:

```bash
./build/Lab1
```

Choose:

```text
15
```

Then choose:

```text
1
```

The program runs on:

```text
Project/portocaliu.bmp
Project/rosu.bmp
Project/galben.bmp
Project/verde.bmp
Project/albastru.bmp
Project/alb.bmp
```

For each image:

- debug windows are opened;
- the final result is displayed;
- the detected face component is printed;
- the 3x3 color matrix is printed.

Press any key in the OpenCV window to move to the next image.

### Live camera test

Run:

```bash
./build/Lab1
```

Choose:

```text
15
```

Then choose:

```text
2
```

To exit live mode, press:

```text
q
```

or:

```text
Esc
```

---

## Common Problems and Fixes

### OpenCV is not found during build

Symptom:

```text
Could not find OpenCV
```

Possible fixes:

- check that OpenCV is installed;
- check `OpenCV_DIR` in `CMakeLists.txt`;
- on macOS, check whether `/opt/homebrew/opt/opencv/lib/cmake/opencv4` exists;
- rerun `cmake ..` inside the `build` directory.

### Camera does not open

Possible fixes:

- check camera permissions in the operating system;
- on macOS, allow camera access for Terminal or CLion;
- close other applications that may be using the camera;
- try an external camera;
- run static image mode first to verify that the algorithm itself works.

### No face is detected

Possible causes:

- lighting is too weak;
- the cube is too small in the image;
- the hand covers too much of the cube;
- the face is too tilted;
- colors are washed out by light;
- the background has similar colors.

Things to try:

- move the cube closer to the camera;
- rotate the cube so the front face is clearer;
- use more even lighting;
- inspect the `H`, `S`, and `V` windows;
- inspect `Colored mask` and `Clean colored mask`.

### White is confused with reflections

The algorithm tries to avoid this by using:

- low saturation requirement;
- high value requirement;
- majority voting;
- central-cell classification;
- ignoring very bright, low-saturation highlight-like pixels.

If the issue remains:

- change the angle relative to the light;
- avoid direct light on the sticker;
- inspect `ROI warped face`.

### Red and orange are confused

Red and orange are close in HSV, so RGB rules are also used.

Red must have strong `R` dominance.

Orange must have dominant `R`, but also enough `G`.

If confusion still happens:

- check the lighting;
- check that the ROI contains only the front face;
- check that the side face is not included in the ROI;
- adjust thresholds in `isOrangeRubikPixel`, `isRedRubikPixel`, or `classifyStickerPixel`.

### The grid looks shifted

The final grid is built in the warped 300x300 ROI and then mapped back to the original image.

If the grid looks shifted:

- first check the red outline of the detected face;
- if the face outline is wrong, the issue is face detection;
- if the face outline is good, inspect `ROI warped face`;
- if the warped face is good, inspect `ROI sticker candidates`.

---

## Code Style Notes

The code is written as a laboratory project, not as a general-purpose library.

The project intentionally keeps:

- simple functions;
- small structs;
- explicit logic;
- visible thresholds;
- easy-to-follow debug windows;
- integration with the existing laboratory menu.

The code does not use large classes, templates, or complex abstractions because the main goal is to make the image processing steps understandable.

---

## Summary

Peek-a-Cube detects the main face of a Rubik's cube using color segmentation, morphology, and connected components. For the white face, it uses Canny on the brightness channel. After the face is detected, the algorithm crops and warps only that region, then detects and classifies stickers only inside that ROI.

This approach is more stable than a global 3x3 grid search because it reduces the risk of including the hand, background, or a side face of the cube.

The main function remains:

```cpp
processRubikFrame(...)
```

For static testing:

```text
15 - Project
1 - Run on images from Project/ folder
```

For live camera testing:

```text
15 - Project
2 - Open camera and detect live
```
