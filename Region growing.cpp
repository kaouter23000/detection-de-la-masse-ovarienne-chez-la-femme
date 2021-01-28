#include <iostream>
#include <stack>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <stdio.h>
#include "tinyxml2.h"
#include <string>

using std::cout;
using std::endl;
using std::stack;
using namespace tinyxml2;

void growing(cv::Mat& image, cv::Mat& mask, cv::Point seed, int threshold);
int moyenne(cv::Mat& image, cv::Point seed);
int threshold = 0;
const uchar max_region_num = 100;
const double min_region_area_factor = 0.01;

int dilation_size = 3;
const cv::Point PointShift2D[8] =
{
    cv::Point(1, 0),
    cv::Point(1, -1),
    cv::Point(0, -1),
    cv::Point(-1, -1),
    cv::Point(-1, 0),
    cv::Point(-1, 1),
    cv::Point(0, 1),
    cv::Point(1, 1)
};

int main() {

    //Chargement de fichiers T2.jpeg
    cv::String path("./img/*.jpeg");
    std::vector<cv::String> fnom;

    cv::glob(path, fnom, true);
    cout << "Nombre de fichiers: " <<fnom.size() << "\n\n";
    double totalDICE = 0;
    for (size_t k = 0; k < fnom.size(); ++k)
    {
        cout << "Fichier "<< k+1 << "\n";
        cout << fnom[k] << "\n";
        //nom de fichier JPG
        std::string nomfichierJPG = fnom[k];
        //nom de fichier XML
        std::string nomfichierXML = "./xml\\"+ nomfichierJPG.substr(6, nomfichierJPG.length() - 11)+".xml";
        cout << nomfichierXML << "\n";
        const char* constXML = nomfichierXML.c_str();

        //Lecture du fichier XML
        const char* x;
        const char* y;
        float coorX, coorY;
        XMLDocument doc;
        doc.LoadFile(constXML);

        std::vector<cv::Point> contour;
        for (const tinyxml2::XMLElement* child = doc.FirstChildElement("ImageAnnotationCollection")
            ->FirstChildElement("imageAnnotations")->FirstChildElement("ImageAnnotation")
            ->FirstChildElement("markupEntityCollection")->FirstChildElement("MarkupEntity")
            ->FirstChildElement("twoDimensionSpatialCoordinateCollection")
            ->FirstChildElement("TwoDimensionSpatialCoordinate");
            child;
            child = child->NextSiblingElement())
        {
            x = child->FirstChildElement("x")->ToElement()->Attribute("value");
            y = child->FirstChildElement("y")->ToElement()->Attribute("value");
            coorX = std::stod(x);
            coorY = std::stod(y);
            contour.push_back(cv::Point(coorX, coorY)); 
        }

        //Find the center of a Blob (Centroid)
        cv::Moments m = cv::moments(contour, true);
        cv::Point centroid(m.m10 / m.m00, m.m01 / m.m00);
        cout << "Barycentre: (" <<centroid.x << "," << centroid.y <<")"<< endl;

        //Lecture de fichier JPEG
        cv::Mat image = cv::imread(nomfichierJPG);//atribut
        assert(!image.empty());

        //Fichier image_XML
        const cv::Point* pts[1] = { (const cv::Point*) cv::Mat(contour).data };
        int npt[] = { contour.size() };
        cv::Mat image_xml = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);//atribut

        cv::fillPoly(image_xml, pts, npt, 1, cv::Scalar(255, 255, 255), 8);
        cv::circle(image_xml, centroid, 2, cv::Scalar(128, 0, 0), -1);
        cv::imshow("Image XML", image_xml);

        cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Image", image);
        cv::waitKey(0);

        //Algorithme de croissance de région
        int min_region_area = int(min_region_area_factor * image.cols * image.rows);
        uchar padding = 1;
        // "mask" records current region, always use "1" for padding
        cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);

        threshold = moyenne(image, cv::Point(centroid.x, centroid.y));
        cout << "Moyenne threshold: " << threshold << "\n";

        //traversal the whole image, apply "seed grow" in undetermined pixels
        for (int x = centroid.x; x < image.cols; ++x) {
            for (int y = centroid.y; y < image.rows; ++y) {
                growing(image, mask, cv::Point(x, y), threshold);
                int mask_area = (int)cv::sum(mask).val[0];  // calculate area of the region that we get in "seed grow"
                if (mask_area > 1000) {
                    //Post-traitement de morphologie mathématique opération de fermeture)
                    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                    cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                    cv::Point(dilation_size, dilation_size));

                    dilate(mask, mask, element);
                    cv::imshow("Region growing", mask * 255);
                    cv::waitKey(0);
                    goto tchao;
                    if (++padding > max_region_num) {
                        printf("run out of max_region_num...");
                        return -1;
                    }
                }
            }
        }
       
        tchao:;
        //DICE index = 2∗bothAB | A | +| B |= 2∗bothAB / onlyA + onlyB + 2∗bothAB
        int mask1 = 0, mask0 = 0, xml0 = 0, xml1 = 0, bothAB = 0, onlyA = 0, onlyB = 0;
        int size = image_xml.cols * image_xml.rows;
        
        for (int im = 0; im < size; im++) {
            if (image_xml.at<cv::Vec3b>(im)[0] == 255)
                xml1++;
            if (image_xml.at<cv::Vec3b>(im)[0] == 0)
                xml0++;
            if (mask.at<uchar>(im) == 1)
                mask1++;
            if (mask.at<uchar>(im) == 0)
                mask0++;
            if (image_xml.at<cv::Vec3b>(im)[0] == 255 && (mask.at<uchar>(im) == 1)) //bothAB
                bothAB++;
            if (image_xml.at<cv::Vec3b>(im)[0] == 255 && (mask.at<uchar>(im) == 0)) //onlyA
                onlyA++;
            if (image_xml.at<cv::Vec3b>(im)[0] == 0 && (mask.at<uchar>(im) == 1)) //onlyB
                onlyB++;
        }

        double DICE = (float)(2 * bothAB) / (onlyA + onlyB + 2 * bothAB);
        //cout << "Total pixels: " << image_xml.cols * image_xml.rows << "\n";
        //cout << "XML pixel (255,255,255): " << xml1 << "\n";
        //cout << "XML pixel (0,0,0): " << xml0 << "\n";
        //cout << "Reg growing pixel (255,255,255): " << mask1 << "\n";
        //cout << "Reg growing pixel (0,0,0): " << mask0 << "\n\n";

        //cout << "Union pixel (255,255,255): " << bothAB << "\n";
        //cout << "XML juste pixel (255,255,255): " << onlyA << "\n";
        //cout << "Reg growing juste pixel (255,255,255): " << onlyB << "\n";
        cout << "Coefficient DICE: " << DICE << "\n\n";
        totalDICE += DICE;
    }
    cout << "Moyenne coefficient DICE: " << totalDICE / fnom.size() << "\n";

    return 0;

}

int moyenne(cv::Mat& image, cv::Point seed) {
    double th = 0;
    int count=0;
    for(int x= seed.x-25;x<= seed.x+25;x++)
        for (int y = seed.y-25; y <= seed.y+25; y++){
            if (x == seed.x && y == seed.y) {
                continue;
            }
            else {
                th += abs(int(image.at<cv::Vec3b>(seed)[0] - image.at<cv::Vec3b>(x, y)[0]));
                //cout << "image.at<cv::Vec3b>("<<seed.x<<","<<seed.y<<")[0]: " << int(image.at<cv::Vec3b>(seed)[0]) << "\n";
                //cout << "image.at<cv::Vec3b>("<<x<<","<< y<<"): " << int(image.at<cv::Vec3b>(x, y)[0]) << "\n";
                count++;
                }
            }
    //cout << "count: " << count << "\n";
    return th/ count;
}

void growing(cv::Mat& image, cv::Mat& mask, cv::Point seed, int threshold) {
    /* apply "seed grow" in a given seed
     * Params:
     *   src: source image
     *   mask: a matrix records the region found in current "seed grow"
     */
    stack<cv::Point> point_stack;
    point_stack.push(seed);

    while (!point_stack.empty()) {
        cv::Point center = point_stack.top();
        //cout<<"point_stack.top(): "<< point_stack.top()<<"\n";
        mask.at<uchar>(center) = 1;
        point_stack.pop();

        for (int i = 0; i < 8; ++i) {
            cv::Point estimating_point = center + PointShift2D[i];
            if (estimating_point.x < 0
                || estimating_point.x > image.cols - 1
                || estimating_point.y < 0
                || estimating_point.y > image.rows - 1) {
                // estimating_point should not out of the range in image
                continue;
            }
            else 
            {
                //uchar delta = (uchar)abs(src.at<uchar>(center) - src.at<uchar>(estimating_point));
                // delta = (R-R')^2 + (G-G')^2 + (B-B')^2
                int delta = int(pow(image.at<cv::Vec3b>(center)[0] - image.at<cv::Vec3b>(estimating_point)[0], 2)
                    + pow(image.at<cv::Vec3b>(center)[1] - image.at<cv::Vec3b>(estimating_point)[1], 2)
                    + pow(image.at<cv::Vec3b>(center)[2] - image.at<cv::Vec3b>(estimating_point)[2], 2));
                if (mask.at<uchar>(estimating_point) == 0 && delta < threshold) {
                    mask.at<uchar>(estimating_point) = 1;  
                    point_stack.push(estimating_point);
                }
            }
        }
    }
}