#include "EnlightenGAN.h"
#include "EnlightenGAN.h"
#include <iostream>
#include <vector>
//#include "cuda_provider_factory.h"
#include "onnxruntime_session_options_config_keys.h"

EnlightenGAN::EnlightenGAN(const TCharString &model_path) : 
    model_path_(model_path), env_(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Default")) {
    createSession();

    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_input_nodes = session_.GetInputCount();
    size_t num_output_nodes = session_.GetOutputCount();
    
    for (int i = 0; i < num_input_nodes; ++i) {
        char *input_name = session_.GetInputName(i, allocator);
        input_names_.emplace_back(input_name);
    }
    
    for (int i = 0; i < num_output_nodes; ++i) {
        char *output_name = session_.GetOutputName(i, allocator);
        output_names_.emplace_back(output_name);
    }
}

EnlightenGAN::~EnlightenGAN() {
}

void EnlightenGAN::predict(cv::Mat image, cv::OutputArray enlighted) {
    cv::Mat imagef32(image.size(), CV_32FC3, cv::Scalar::all(0)), gray(image.size(), CV_32FC1, cv::Scalar(0));
    const uchar *row_input = image.data, *op_input;
    float *row_image = reinterpret_cast<float *>(imagef32.data), *row_gray = reinterpret_cast<float *>(gray.data), *op_image, vr, vg, vb;
    int ss = image.cols * image.rows, offset;
    for (int s = 0; s < ss; ++s) {
        offset = s * 3;
        op_input = row_input + offset;
        op_image = row_image + offset;

        vr = (static_cast<float>(op_input[2]) / 255.0 - 0.5) / 0.5;
        vg = (static_cast<float>(op_input[1]) / 255.0 - 0.5) / 0.5;
        vb = (static_cast<float>(op_input[0]) / 255.0 - 0.5) / 0.5;

        op_image[0] = vr;
        op_image[1] = vg;
        op_image[2] = vb;

        row_gray[s] = 1.0 - ((vr + 1) * 0.299 + (vg + 1) * 0.587 + (vb + 1) * 0.114) / 2.0;
    }

    const int height = imagef32.rows;
    const int width = imagef32.cols;
    
    std::vector<float> image_tensor_data, gray_tensor_data;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPUInput);

    Ort::Value image_tensor = mat2Tensor(imagef32, memory_info, image_tensor_data);
    Ort::Value gray_tensor = mat2Tensor(gray, memory_info, gray_tensor_data);

    //assert(image_tensor.IsTensor());
    //assert(gray_tensor.IsTensor());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.emplace_back(std::move(image_tensor));
    ort_inputs.emplace_back(std::move(gray_tensor));
    
    auto outputs = session_.Run(Ort::RunOptions{ nullptr }, input_names_.data(), ort_inputs.data(), ort_inputs.size(), output_names_.data(), output_names_.size());
    float *out_data = outputs.front().GetTensorMutableData<float>();

    enlighted.create(height, width, CV_8UC3);

    cv::Mat _enlighted = enlighted.getMat(-1);
    int i = 0;
    for (int c = 2; c >= 0; --c) {
        for (int h = 0; h < height; ++h) {
            uchar *row_ptr = _enlighted.ptr<uchar>(h);
            for (int w = 0; w < width; ++w) {
                row_ptr[w * 3 + c] = cv::saturate_cast<uchar>(out_data[i++] * 127.5 + 127.5);
            }
        }
    }
}

void EnlightenGAN::createSession() {
    Ort::SessionOptions session_options;
    //Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
    session_ = Ort::Session(env_, model_path_.c_str(), session_options);
}

Ort::Value EnlightenGAN::mat2Tensor(cv::Mat imagef32, const Ort::MemoryInfo &memory_info_handler, std::vector<float> &tensor_data_handler) {
    const int h = imagef32.rows;
    const int w = imagef32.cols;
    const int c = imagef32.channels();
    const size_t elements_per_channel = h * w;
    const size_t elements = elements_per_channel * c;
    std::vector<int64_t> dims({ 1,c,h,w });
    if (c > 1) {
        std::vector<cv::Mat> channels;
        cv::split(imagef32, channels);
        for (cv::Mat &chn : channels) {
            float *pdata = reinterpret_cast<float *>(chn.data);
            tensor_data_handler.insert(tensor_data_handler.end(), pdata, pdata + elements_per_channel);
        }
    } else {
        float *pdata = reinterpret_cast<float*>(imagef32.data);
        tensor_data_handler.insert(tensor_data_handler.end(), pdata, pdata + elements);
    }
    return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_data_handler.data(), elements, dims.data(), dims.size());
}
 



