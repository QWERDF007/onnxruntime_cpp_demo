#pragma once

#include <opencv2/opencv.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/ops/const_op.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>


class __declspec(dllexport) DeepLabv3Plus {
public:
	DeepLabv3Plus(std::string &model_path);
	~DeepLabv3Plus();

	void predict(const cv::Mat &image, cv::OutputArray segmentation);

private:

	tensorflow::Session *session_;
	tensorflow::Status status_;
};