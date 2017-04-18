// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4

#include <abc.h>
#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace cv;

///////////////////////////////////////////////////////////////////////////////////
/// brief The Detector class
struct det_box{
    int x_start;
    int y_start;
    int x_end;
    int y_end;
    double s;
    double c;
};

struct test_histogram{
    int s_0;
    int s_1;
    int s_2;
    int s_3;
    int s_4;
    int s_5;
    int s_6;
    int s_7;
    int s_8;
    int s_9;
    int s_10;
    int s_11;
    int s_12;
    int s_13;
    int s_14;
    int s_15;
    int s_16;
    int s_17;
    int s_18;
    int s_19;
};

int max(int x ,int y)
{
    return (x>y?x:y);
}
int min(int x ,int y)
{
    return (x<y?x:y);
}
int max_score(std::vector<det_box>& input)
{
    double tmp =-1000;
    int idx=0;
    std::vector<double>score;
    for(size_t ii=0;ii<input.size();ii++)
    {
        score.push_back(input[ii].s);
    }
    for(size_t ii=0;ii<score.size();ii++)
    {
        if(tmp<score[ii])
        {
            tmp=score[ii];
            idx=ii;
        }
    }
    return idx;
}

std::vector<det_box>  nms(std::vector<det_box> & boxes,double threshold,int imgArea)
{
    std::vector<int>x1,x2,y1,y2;
    std::vector<double> score,area;
    for(size_t ii=0;ii<boxes.size();ii++)
    {
        x1.push_back(boxes[ii].x_start);
        x2.push_back(boxes[ii].x_end);
        y1.push_back(boxes[ii].y_start);
        y2.push_back(boxes[ii].y_end);
        score.push_back(boxes[ii].s);
        area.push_back((boxes[ii].x_end-boxes[ii].x_start+1)*(boxes[ii].y_end-boxes[ii].y_start+1));
    }
    std::vector<det_box> result,tmp,cache;
    tmp = boxes;
    int index=0;std::vector<double> /*xx1,xx2,yy1,yy2,*/interArea;
    int width=0,height=0;
    while(!tmp.empty())
    {
        index = max_score(tmp);
        result.push_back(tmp[index]);

        int s_x1 = x1[index];int s_y1 = y1[index];
        int s_x2 = x2[index];int s_y2 = y2[index];
        for(size_t ii=0;ii<tmp.size();ii++)
        {
            //            xx1.push_back(max(s_x1,tmp[index].x_start));
            //            xx2.push_back(min(s_x2,tmp[index].x_end));
            //            yy1.push_back(max(s_y1,tmp[index].y_start));
            //            yy2.push_back(min(s_y2,tmp[index].y_end));
            width = max(0.0,(min(s_x2,tmp[ii].x_end)-max(s_x1,tmp[ii].x_start)+1));
            height= max(0.0,(min(s_y2,tmp[ii].y_end)-max(s_y1,tmp[ii].y_start)+1));
            interArea.push_back((double)(width*height));
        }
        //std::vector<double> ratio;
        for(size_t ii=0;ii<tmp.size();ii++)
        {
            //            ratio.push_back(interArea[ii]/(area[index]+area[ii]-interArea));
            if(area[ii]>imgArea/3.5)
            {
                if((interArea[ii]/min(area[index],area[ii])<threshold)&&ii!=index)
                    cache.push_back(tmp[ii]);

            }else
            {
                if((interArea[ii]/(area[index]+area[ii]-interArea[ii])<threshold)&&ii!=index)
                    cache.push_back(tmp[ii]);
            }
        }
        tmp=cache;
        cache.clear();
        interArea.clear();

    }
    return result;
}

///////////////////////////////////////////////////////////////////////////////////

DEFINE_string(mean_file, "",
              "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "104,117,123",
              "If specified, can be one value or can be same as image channels"
              " - would subtract from the corresponding channel). Separated by ','."
              "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
              "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
              "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.5,
              "Only store detections with score higher than the threshold.");


struct st{
    float score;
    int label;
    int index;
};

static std::string det_string[] =
{
    "background",
    "chelian",
    "sanjiaojia",
    "anquandai",
    "chejia",
    "xingshizheng"
};

bool score_sort(vector<float> v1, vector<float> v2)
{
    return v1[2] > v2[2];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main()
{
    const string& model_file = "/home/em-gkj/caffe-ssd/models/VGGNet/VOCmy/SSD_300x300/deploy.prototxt";
    const string& weights_file = "/home/em-gkj/caffe-ssd/models/VGGNet/VOCmy/SSD_300x300/VGG_VOCmy_SSD_300x300_iter_100000.caffemodel";
    //std::ifstream infile("/home/em-gkj/devdata/train-data/chelian/chelian_sanjiaojia/list.txt");
    std::ifstream infile("/home/em-gkj/devdata/train-data/chelian/anquandai/list.txt");
    //std::ifstream infile("/home/em-gkj/devdata/train-data/chelian/xingshizheng/list.txt");
    const string& mean_file = FLAGS_mean_file;

    const string& mean_value = FLAGS_mean_value;
    const string& file_type = FLAGS_file_type;
    const string& out_file = FLAGS_out_file;
    //const float confidence_threshold = FLAGS_confidence_threshold;

    // Initialize the network.
    Detector detector(model_file, weights_file, mean_file, mean_value);

    // Set the output mode.
    std::streambuf* buf = std::cout.rdbuf();
    std::ofstream outfile;
    if (!out_file.empty())
    {
        outfile.open(out_file.c_str());
        if (outfile.good())
        {
            buf = outfile.rdbuf();
        }
    }

    ostream out(buf);
    string file;

    struct test_histogram sanjiaojia={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    int cnt = 0;
    while (infile >> file)
    {
        cnt++;
        cout << cnt << endl;

        if (file_type == "image")
        {
            Mat img = imread(file, -1);

            long long time_begin;
            long long time_end;
            time_begin = cvGetTickCount();

            vector<vector<float> > detections = detector.Detect(img);

            time_end = cvGetTickCount();
            printf("\t time = %f\n", (time_end - time_begin) / cvGetTickFrequency() / 1000000);

            std::sort(detections.begin(), detections.end(), score_sort);

            /* Print the detection results. */
            for (size_t i = 0; i < detections.size() && i<5; ++i)
            {
                Mat show = img.clone();
                const vector<float>& d = detections[i];
                // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                CHECK_EQ(d.size(), 7);
                const float score = d[2];

                int label = int(d[1]);
                std::cout << det_string[label] << " " << score << std::endl;
                if(label == 3)
                {
                    if(score>=0 && score<0.05)
                    {
                        sanjiaojia.s_0++;
                    }
                    else if(score>=0.05 && score<0.1)
                    {
                        sanjiaojia.s_1++;
                    }
                    else if(score>=0.1 && score<0.15)
                    {
                        sanjiaojia.s_2++;
                    }
                    else if(score>=0.15 && score<0.2)
                    {
                        sanjiaojia.s_3++;
                    }
                    else if(score>=0.2 && score<0.25)
                    {
                        sanjiaojia.s_4++;
                    }
                    else if(score>=0.25 && score<0.3)
                    {
                        sanjiaojia.s_5++;
                    }
                    else if(score>=0.3 && score<0.35)
                    {
                        sanjiaojia.s_6++;
                    }
                    else if(score>=0.35 && score<0.4)
                    {
                        sanjiaojia.s_7++;
                    }
                    else if(score>=0.4 && score<0.45)
                    {
                        sanjiaojia.s_8++;
                    }
                    else if(score>=0.45 && score<0.5)
                    {
                        sanjiaojia.s_9++;
                    }
                    else if(score>=0.5 && score<0.55)
                    {
                        sanjiaojia.s_10++;
                    }
                    else if(score>=0.55 && score<0.60)
                    {
                        sanjiaojia.s_11++;
                    }
                    else if(score>=0.60 && score<0.65)
                    {
                        sanjiaojia.s_12++;
                    }
                    else if(score>=0.65 && score<0.7)
                    {
                        sanjiaojia.s_13++;
                    }
                    else if(score>=0.7 && score<0.75)
                    {
                        sanjiaojia.s_14++;
                    }
                    else if(score>=0.75 && score<0.8)
                    {
                        sanjiaojia.s_15++;
                    }
                    else if(score>=0.8 && score<0.85)
                    {
                        sanjiaojia.s_16++;
                    }
                    else if(score>=0.85 && score<0.9)
                    {
                        sanjiaojia.s_17++;
                    }
                    else if(score>=0.9 && score<0.95)
                    {
                        sanjiaojia.s_18++;
                    }
                    else if(score>=0.95 && score<=1)
                    {
                        sanjiaojia.s_19++;
                    }
                    rectangle(show,cv::Rect((d[3] * img.cols),(d[4] * img.rows),(d[5] * img.cols -d[3] * img.cols),(d[6] * img.rows-d[4] * img.rows)),cv::Scalar(0,0,255));
                    imshow("det",show);
                    waitKey(0);
                    break;
                }
                //cv::rectangle(show,cv::Rect((d[3] * img.cols),(d[4] * img.rows),(d[5] * img.cols -d[3] * img.cols),(d[6] * img.rows-d[4] * img.rows)),cv::Scalar(0,0,255));
                
                //                cv::imshow("det",show);
                //                cv::waitKey(0);
            }
        }
        else
        {
            LOG(FATAL) << "Unknown file_type: " << file_type;
        }
    }
    int all = sanjiaojia.s_0+sanjiaojia.s_1+sanjiaojia.s_2+sanjiaojia.s_3+sanjiaojia.s_4
            +sanjiaojia.s_5+sanjiaojia.s_6+sanjiaojia.s_7+sanjiaojia.s_8+sanjiaojia.s_9
            +sanjiaojia.s_10+sanjiaojia.s_11+sanjiaojia.s_12+sanjiaojia.s_13+sanjiaojia.s_14
            +sanjiaojia.s_15+sanjiaojia.s_16+sanjiaojia.s_17+sanjiaojia.s_18+sanjiaojia.s_19;
    cout<<sanjiaojia.s_0<<"-"<<sanjiaojia.s_1<<"-"<<sanjiaojia.s_2<<"-"<<sanjiaojia.s_3<<"-"
          <<sanjiaojia.s_4<<"-"<<sanjiaojia.s_5<<"-"<<sanjiaojia.s_6<<"-"<<sanjiaojia.s_7<<"-"
            <<sanjiaojia.s_8<<"-"<<sanjiaojia.s_9<<"-"<<sanjiaojia.s_10<<"-"<<sanjiaojia.s_11
           <<"-"<<sanjiaojia.s_12<<"-"<<sanjiaojia.s_13<<"-"<<sanjiaojia.s_14<<"-"<<sanjiaojia.s_15
          <<"-"<<sanjiaojia.s_16<<"-"<<sanjiaojia.s_17<<"-"<<sanjiaojia.s_18<<"-"<<sanjiaojia.s_19<< endl ;
    cout<<all <<endl;
    waitKey(0);
    return 0;
}
#else
int main(int argc, char** argv)
{
    LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
