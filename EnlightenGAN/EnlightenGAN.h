#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using TCharString = std::basic_string<ORTCHAR_T>;


class __declspec(dllexport) EnlightenGAN {
public:
    EnlightenGAN(const TCharString &model_path);

    ~EnlightenGAN();

    void predict(cv::Mat image, cv::OutputArray enlighted);

private:

    std::vector<const char *> input_names_;
    std::vector<const char *> output_names_;
    Ort::Env &env_;
    const TCharString model_path_;
    Ort::Session session_{ nullptr };

    void createSession();

    Ort::Value mat2Tensor(cv::Mat imagef32, const Ort::MemoryInfo &memory_info_handler, std::vector<float> &tensor_data_handler);
};