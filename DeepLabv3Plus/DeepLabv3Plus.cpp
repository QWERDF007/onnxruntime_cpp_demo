#include "DeepLabv3Plus.h"
#include <iostream>


DeepLabv3Plus::DeepLabv3Plus(std::string &model_path) 
{
    tensorflow::GraphDef graph_def;
    tensorflow::SessionOptions session_options;

    status_ = tensorflow::NewSession(session_options, &session_);
    if (!status_.ok()) {
        throw std::runtime_error(status_.ToString());
    }

    status_ = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_path.c_str(), &graph_def);
    if (!status_.ok()) {
        throw std::runtime_error(status_.ToString());
    }

    status_ = session_->Create(graph_def);
    if (!status_.ok()) {
        throw std::runtime_error(status_.ToString());
    }

}

DeepLabv3Plus::~DeepLabv3Plus() {
    if (session_ != nullptr)
        delete session_;
}

void DeepLabv3Plus::predict(const cv::Mat &image, cv::OutputArray segmentation) {
	const int height = image.rows;
	const int width = image.cols;
    segmentation.create(image.size(), CV_8UC1);
    cv::Mat res = segmentation.getMat(-1);
	tensorflow::Tensor tinput(tensorflow::DT_UINT8, tensorflow::TensorShape({ 1, height, width, 3 }));
	auto inputTensorMapped = tinput.tensor<uchar, 4>();
	for (int r = 0; r < height; ++r) {
		const uchar *dataRow = image.ptr<uchar>(r);
		for (int c = 0; c < width; ++c) {
			const uchar *dataPixel = dataRow + (c * 3);
			inputTensorMapped(0, r, c, 0) = dataPixel[2];
			inputTensorMapped(0, r, c, 1) = dataPixel[1];
			inputTensorMapped(0, r, c, 2) = dataPixel[0];
		}
	}

	std::vector<tensorflow::Tensor> outputs;
	tensorflow::Status status_run = session_->Run({ { "ImageTensor:0", tinput } }, { "SemanticPredictions:0" }, { }, &outputs);
	if (!status_run.ok()) {
		throw std::runtime_error(status_run.ToString());
	}

	tensorflow::Tensor outres = outputs[0];
	auto output = outres.tensor<long long, 3>();
	for (int r = 0; r < height; ++r) {
		uchar *res_row = res.ptr<uchar>(r);
		for (int c = 0; c < width; ++c) {
			res_row[c] = static_cast<uchar>(output(0, r, c));
		}
	}
}


