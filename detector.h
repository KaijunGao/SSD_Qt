#ifndef DETECTOR_H
#define DETECTOR_H
#include <abc.h>
class Detector
{
public:
    Detector(const std::string& model_file,
             const std::string& weights_file,
             const std::string& mean_file,
             const std::string& mean_value);

    std::vector<std::vector<float> > Detect(const cv::Mat& img);
private:
    void SetMean(const std::string& mean_file, const std::string& mean_value);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);
private:
    std::shared_ptr<caffe::Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
};

#endif // DETECTOR_H
