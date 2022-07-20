#include "EnlightenGAN/EnlightenGAN.h"
#include "FBAMatting/FBAMatting.h"
#include "DeepLabv3Plus/DeepLabv3Plus.h"

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include <codecvt>
#include <chrono>

cv::Rect EnlightenPadTransform(cv::Mat image, cv::OutputArray outImage, int dsize = 800) {
    double scale = static_cast<double>(dsize) / std::max(image.cols, image.rows);
    int h = std::ceil(image.rows * scale);
    int w = std::ceil(image.cols * scale);
    cv::Rect r(0, 0, w, h);
    cv::Mat tmp;
    outImage.create(dsize, dsize, image.type());
    cv::Mat _out = outImage.getMat(-1);
    cv::resize(image, tmp, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);
    tmp.copyTo(_out(r));
    return r;
}


cv::Rect MattingPadTransform(cv::Mat image, cv::Mat trimap, cv::OutputArray outImage, cv::OutputArray outTrimap, int dsize=512) {
    double scale = static_cast<double>(dsize) / std::max(image.cols, image.rows);
    int h = std::ceil(image.rows * scale);
    int w = std::ceil(image.cols * scale);
    cv::Rect r(0, 0, w, h);
    cv::Mat tmpImage, tmpTrimap;
    outImage.create(dsize, dsize, image.type());
    outTrimap.create(dsize, dsize, trimap.type());
    cv::Mat _outImage = outImage.getMat(-1);
    cv::Mat _outTrimap = outTrimap.getMat(-1);
    cv::resize(image, tmpImage, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);
    cv::resize(trimap, tmpTrimap, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
    tmpImage.copyTo(_outImage(r));
    tmpTrimap.copyTo(_outTrimap(r));
    return r;
}


cv::Rect SegmentPadTransform(cv::Mat image, cv::OutputArray outImage, int dsize = 513) {
    double scale = static_cast<double>(dsize) / std::max(image.cols, image.rows);
    int h = std::ceil(image.rows * scale);
    int w = std::ceil(image.cols * scale);
    cv::Rect r(0, 0, w, h);
    cv::Mat tmp;
    outImage.create(dsize, dsize, image.type());
    cv::Mat _out = outImage.getMat(-1);
    cv::resize(image, tmp, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);
    tmp.copyTo(_out(r));
    return r;
}


inline std::wstring to_wide_string(const std::string &input) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    return converter.from_bytes(input);
}


void Segmentation2Trimap(cv::Mat trimap) 
{
    static const uchar mapper[3] = { 0, 128, 255 };
    size_t len = trimap.rows * trimap.cols;
    uchar *pdata = trimap.data;
    for (size_t i = 0; i < len; ++i) {
        pdata[i] = mapper[pdata[i]];
    }
}

void MattingDemo(std::string &image_path, std::string &trimap_path, std::string &matting_model, std::string &segment_model) 
{
    if (!matting_model.empty() && !segment_model.empty()) {
        if (!image_path.empty()) {
            spdlog::info("Loading Segment Model: {}", segment_model);
            DeepLabv3Plus SegmentPredictor(segment_model);
            FBAMatting FBAMattingPredictor(to_wide_string(matting_model));
            cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
            cv::Mat padded_image, trimap;
            cv::Rect r = SegmentPadTransform(image, padded_image);
            SegmentPredictor.predict(padded_image, trimap);
            cv::resize(trimap(r), trimap, image.size(), 0, 0, cv::INTER_NEAREST);

            Segmentation2Trimap(trimap);
            cv::Mat padded_trimap;
            r = MattingPadTransform(image, trimap, padded_image, padded_trimap);
            cv::Mat fg, alpha;
            FBAMattingPredictor.predict(padded_image, padded_trimap, fg, alpha);
            cv::resize(fg(r), fg, image.size(), 0, 0, cv::INTER_CUBIC);
            cv::resize(alpha(r), alpha, image.size(), 0, 0, cv::INTER_CUBIC);
        } else {
            spdlog::error("File not found: {}", image_path);
        }
    } else if (!matting_model.empty()) {
        if (!trimap_path.empty() && !image_path.empty()) {
            FBAMatting FBAMattingPredictor(to_wide_string(matting_model));
            cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
            cv::Mat trimap = cv::imread(trimap_path, cv::IMREAD_GRAYSCALE);
            cv::Mat padded_image, padded_trimap;
            cv::Rect r = MattingPadTransform(image, trimap, padded_image, padded_trimap);
            cv::Mat fg, alpha;
            FBAMattingPredictor.predict(padded_image, padded_trimap, fg, alpha);
            cv::resize(fg(r), fg, image.size(), 0, 0, cv::INTER_CUBIC);
            cv::resize(alpha(r), alpha, image.size(), 0, 0, cv::INTER_CUBIC);
        } else {
            if (image_path.empty())
                spdlog::error("File not found: {}", image_path);
            if (trimap_path.empty())
                spdlog::error("File not found: {}", trimap_path);
        }
    } else {
        spdlog::info("File nout found");
    }
}


int main(int argc, char **argv) 
{
    
    try {
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S,%e] [%n] [%^%L%$] [%t] %v");
        cxxopts::Options options("CredCpuTest", "Cred Cpu Test");
        options.add_options()
            ("m,matting_model", "matting model file", cxxopts::value<std::string>()->default_value(""))
            ("e,enlight_model", "enlight model file", cxxopts::value<std::string>()->default_value(""))
            ("s,segment_model", "segment model file", cxxopts::value<std::string>()->default_value(""))
            ("i,image", "input image", cxxopts::value<std::string>()->default_value(""))
            ("t,trimap", "input trimap", cxxopts::value<std::string>()->default_value(""))
            ("n,num", "repeat times", cxxopts::value<int>()->default_value("10"))
            ("h,help", "print usage");
           
        auto args = options.parse(argc, argv);
        if (args.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        std::string enlight_model = args["enlight_model"].as<std::string>();
        std::string matting_model = args["matting_model"].as<std::string>();
        std::string segment_model = args["segment_model"].as<std::string>();
        int n = args["num"].as<int>();
        spdlog::info("enlight model: {}", enlight_model);
        spdlog::info("matting model: {}", matting_model);
        spdlog::info("segment model: {}", segment_model);
        spdlog::info("repeat times: {}", n);

        if ((matting_model == "" || matting_model.empty()) && 
            (enlight_model == "" || enlight_model.empty()) && 
            (segment_model == "" || segment_model.empty())) {
            spdlog::error("no found any model, exit(1).");
            exit(1);
        }

        std::string image_path = args["image"].as<std::string>();
        std::string trimap_path = args["trimap"].as<std::string>();
        cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
        spdlog::info("image size: {}x{}", image.cols, image.rows);
        

        if (matting_model != "" && !matting_model.empty()) {
            cv::Mat trimap = cv::imread(trimap_path, cv::IMREAD_UNCHANGED);
            cv::Mat paddedImage, paddedTrimap;
            cv::Rect r = MattingPadTransform(image, trimap, paddedImage, paddedTrimap);
            cv::Mat fg, alpha;
            FBAMatting predictor(to_wide_string(matting_model).c_str());
            spdlog::info("start run {} times", n);
            std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
            for (int i = 0; i < n; ++i) {
                predictor.predict(paddedImage, paddedTrimap, fg, alpha);
            }
            std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
            spdlog::info("done");
            std::chrono::duration<double> diff = end - start;
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
            spdlog::info("image size: {}x{}", image.cols, image.rows);
            spdlog::info("elapsed time: {} ms, aver: {} ms", elapsed, elapsed / n);
            cv::resize(alpha(r), alpha, image.size());
            cv::resize(fg(r), fg, image.size());
            cv::imshow("alpha", alpha);
            cv::imshow("fg", fg);
        }

        if (enlight_model != "" && !enlight_model.empty()) {
            cv::Mat enlighted, paddedImage;
            cv::Rect r = EnlightenPadTransform(image, paddedImage);
            EnlightenGAN predictor(to_wide_string(enlight_model).c_str());
            spdlog::info("start run {} times", n);
            std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
            for (int i = 0; i < n; ++i) {
                predictor.predict(paddedImage, enlighted);
            }
            std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
            spdlog::info("done");
            std::chrono::duration<double> diff = end - start;
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
            spdlog::info("image size: {}x{}", image.cols, image.rows);
            spdlog::info("elapsed time: {} ms, aver: {} ms", elapsed, elapsed / n);
            cv::resize(enlighted(r), enlighted, image.size());
            cv::imshow("enlighted", enlighted);
        }

        if (segment_model != "" && !segment_model.empty()) {
            cv::Mat segmentation, paddedImage;
            cv::Rect r = SegmentPadTransform(image, paddedImage);
            DeepLabv3Plus predictor(segment_model);
            spdlog::info("start run {} times", n);
            std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
            for (int i = 0; i < n; ++i) {
                predictor.predict(paddedImage, segmentation);
            }
            std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
            spdlog::info("done");
            std::chrono::duration<double> diff = end - start;
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
            spdlog::info("image size: {}x{}", image.cols, image.rows);
            spdlog::info("elapsed time: {} ms, aver: {} ms", elapsed, elapsed / n);
            cv::resize(segmentation(r), segmentation, image.size());
            cv::normalize(segmentation, segmentation, 0, 255, cv::NORM_MINMAX);
            cv::imshow("segmentation", segmentation);
        }

        cv::imshow("image", image);
        cv::waitKey();
    } catch (Ort::Exception &e) {
        std::cout << "Ort Exception: " << e.what() << std::endl;
    } catch (cv::Exception &e) {
        std::cout << "OpenCV Exception: " << e.what() << std::endl;
    } catch (std::runtime_error &e) {
        std::cout << "Runtime Exception: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Unknown Exception: "<< std::endl;
    }
    return 0;
}