#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using TCharString = std::basic_string<ORTCHAR_T>;


class __declspec(dllexport) FBAMatting {
public:
	FBAMatting(const TCharString &model_path);
	~FBAMatting();

	void predict(const cv::Mat &image, const cv::Mat &trimap, cv::OutputArray fg, cv::OutputArray alpha);

private:

	const static int L;
	const static float L1;
	const static float L2;
	const static float L3;

	std::vector<const char *> input_names_;
	std::vector<const char *> output_names_;
	Ort::Env &env_;
	const TCharString model_path_;
	Ort::Session session_{ nullptr };

	void createSession();

	void padTransform(cv::Mat image, cv::Mat trimap, cv::OutputArray outImage, cv::OutputArray outTrimap);

	cv::Mat trimapTransform(const cv::Mat &trimapF32);
	cv::Mat normalizeImage(const cv::Mat &imageF32);
	cv::Mat generatedTrimap(const cv::Mat &trimapF32);
	void getFinalAlpha(const cv::Mat &pred, const cv::Mat &trimaps, cv::OutputArray alpha);
	void getFinalFg(const cv::Mat &image, const cv::Mat &tmpAlpha, const cv::Mat &tmpFg, cv::OutputArray fg);

	Ort::Value mat2Tensor(cv::Mat imagef32, const Ort::MemoryInfo &memory_info_handler, std::vector<float> &tensor_data_handler);
};