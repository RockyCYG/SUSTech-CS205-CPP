#pragma once

#include <opencv2/opencv.hpp>
#include <algorithm>
#include "face_binary_cls.cpp"
#include "matrix.hpp"

void imgToMat(const cv::Mat &img, Matrix<float> &matrix, size_t kernel_size, size_t paddings, size_t stride);

void convResToMat(Matrix<float> &in, Matrix<float> &out, size_t kernel_size, size_t paddings, size_t stride);

void convAndRelu(Matrix<float> &in, conv_param &conv_p, Matrix<float> &out);

void maxPooling(Matrix<float> &in, Matrix<float> &out);

void fullyConnected(Matrix<float> &in, fc_param &fc_p, Matrix<float> &out);