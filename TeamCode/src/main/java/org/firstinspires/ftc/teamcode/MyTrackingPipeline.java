/* Copyright (c) 2017 FIRST. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted (subject to the limitations in the disclaimer below) provided that
 * the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this list
 * of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice, this
 * list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * Neither the name of FIRST nor the names of its contributors may be used to endorse or
 * promote products derived from this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS
 * LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package org.firstinspires.ftc.teamcode;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.List;



public class MyTrackingPipeline extends OpenCvPipeline {
    // ArrayList of centers to track
    private ArrayList<Point> points = new ArrayList<>();
    // Width and height of the last source
    private int prevWidth = -1;
    private int prevHeight = -1;

    public void helperColorDetector(Mat input, Mat newHSV, Scalar lower, Scalar upper, String color, Scalar rectColor) {
        // Turn the newHSV frame into a mask of a detected color
        Mat detection = new Mat();
        Core.inRange(newHSV, lower, upper, detection);
        // Red is on both ends of the HSV spectrum, so I have to merge both detection Mats for red here using bitwise_or
        if (color.equals("Red")) {
            Mat redDetection1 = new Mat();
            Mat redDetection2 = new Mat();
            Core.inRange(newHSV, new Scalar(0, 100, 100), new Scalar(5, 255, 255), redDetection1);
            Core.inRange(newHSV, lower, upper, redDetection2);
            Core.bitwise_or(redDetection1, redDetection2, detection);
        }

        // Contours are outlines of an object in the frame
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        // findContours detects contours in the frame
        Imgproc.findContours(detection, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Here I'm finding the red contour with the largest area so I can track it
        MatOfPoint bigContour = null;
        if (color.equals("Red")) {
            double area = 0;
            for (MatOfPoint contour : contours) {
                double contourArea = Imgproc.contourArea(contour);
                if (contourArea > area) {
                    area = contourArea;
                    bigContour = contour;
                }
            }
        }
        // Here I loop through all the contours, find the center of the largest one, and add it to the points ArrayList
        for (MatOfPoint contour : contours) {
            Rect colorRect = Imgproc.boundingRect(contour);
            if (color.equals("Red") && contour.equals(bigContour)) {
                Point center = new Point(colorRect.x + (colorRect.width / 2.0), colorRect.y + (colorRect.height / 2.0));
                points.add(center);
            }
            // Here, I create rectangles around all the contours and add text saying which color it is
            if (colorRect.area() > 150) { // Only draw rectangles of meaningful size
                Imgproc.rectangle(input, colorRect, rectColor, 1); // red box in RGB
                Imgproc.putText(input, color, new Point(colorRect.x, colorRect.y - 5), Imgproc.FONT_HERSHEY_TRIPLEX, 0.3, new Scalar(255, 255, 255), 1);
            }
        }
    }


    @Override
    public Mat processFrame(Mat input) {
        // currWidth and currHeight are the width and height of the current source
        int currWidth = input.width();
        int currHeight = input.height();

        // Find a change in dimensions, which usually means the source has been changed, then delete all elements from points
        if (currWidth != prevWidth || currHeight != prevHeight) {
            prevWidth = currWidth;
            prevHeight = currHeight;
            points.clear();
        }
        // Convert the input color format from RGB to HSV, HSV is better for color detection, since it separates brightness from hue
        Mat hsv = new Mat();
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);
        helperColorDetector(input, hsv, new Scalar(90, 50, 70), new Scalar(128, 255, 255), "Blue", new Scalar(0, 0, 255));
        helperColorDetector(input, hsv, new Scalar(10, 100, 100), new Scalar(30, 255, 255), "Yellow", new Scalar(255, 255, 0));
        helperColorDetector(input, hsv, new Scalar(160, 100, 100), new Scalar(180, 255, 255), "Red", new Scalar(255, 0, 0));

        // Removing the last line from the trace when the trace becomes too long, limits it to 50 elements
        if (points.size() > 100) {
            points.remove(0);
        }

        // Draws the trace in the frame
        for (int i = 0; i < points.size(); i++) {
            if (i > 0) {
                Point cent = points.get(i - 1);
                Point cent1 = points.get(i);
                Imgproc.line(input, cent, cent1, new Scalar(0, 0, 0), 5);
            }
        }

        return input;
    }
}