#pragma once

#include <cblas.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "face_binary_cls.cpp"
#include "matrix.hpp"

//将输入进来的图片img按照下次卷积中卷积核的属性展开为矩阵乘法的标准形式，输出矩阵为out
void imgToMat(const cv::Mat& img,
              Matrix<float>& out,
              size_t kernel_size,
              size_t paddings,
              size_t stride);

//将每一次卷积后得到的结果矩阵in按照下次卷积中卷积核的属性展开为矩阵乘法的标准形式作为下一次卷积的输入矩阵，输出矩阵为out
void convResToMat(Matrix<float>& in,
                  Matrix<float>& out,
                  size_t kernel_size,
                  size_t paddings,
                  size_t stride);

//对输入进来的矩阵in和卷积核做矩阵乘法，结果为out
void convAndRelu(Matrix<float>& in, conv_param& conv_p, Matrix<float>& out);

//对in做池化操作，结果为out
void maxPooling(Matrix<float>& in, Matrix<float>& out);

//全连接层的矩阵乘法，输入为in以及全连接层的矩阵，结果为out
void fullyConnected(Matrix<float>& in, fc_param& fc_p, Matrix<float>& out);