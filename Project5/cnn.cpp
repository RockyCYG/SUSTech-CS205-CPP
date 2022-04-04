#include "cnn.hpp"

void imgToMat(cv::Mat& img,
              Matrix<float>& out,
              size_t kernel_size,
              size_t paddings,
              size_t stride) {
    assert(img.data != nullptr);
    //先将输出的out矩阵里的数据清空
    delete[] out.getData();
    cv::Mat temp;
    //如果输入图片的尺寸不是128*128，就将它转成128*128的图片
    if (img.rows != 128 || img.cols != 128) {
        cv::resize(img, temp, cv::Size(128, 128));
        img = temp;
    }
    std::vector<cv::Mat> BGR(3);
    cv::split(img, BGR);
    //将img的三个通道分离得到B、G、R
    cv::Mat B = BGR[0];
    cv::Mat G = BGR[1];
    cv::Mat R = BGR[2];
    size_t in_size = img.rows;
    //由公式可以计算卷积后得到的矩阵有多少行(列)
    size_t size = (in_size + 2 * paddings - kernel_size) / stride + 1;
    out.setRows(kernel_size * kernel_size * 3);
    out.setCols(size * size);
    out.getData() = new float[out.getRows() * out.getCols()];
    memset(out.getData(), 0, sizeof(float) * out.getRows() * out.getCols());
    //将输入的矩阵展开，以R、G、B的顺序输出到out矩阵中，并根据padding的值来确定是否要补0，以及kernel_size的值决定补几圈0
    // circle来确定补几圈0
    size_t circle = (kernel_size - 1) / 2;
    size_t col = 0;
    for (size_t i = 0, cnt1 = 0; cnt1 < size; i += stride, cnt1++) {
        for (size_t j = 0, cnt2 = 0; cnt2 < size; j += stride, col++, cnt2++) {
            size_t row = 0;
            for (size_t r = i; r < i + kernel_size; r++) {
                for (size_t c = j; c < j + kernel_size; c++, row++) {
                    if (paddings > 0) {
                        if ((r >= 0 && r <= circle - 1) ||
                            (r >= in_size + circle &&
                             r <= in_size + 2 * circle - 1) ||
                            (c >= 0 && c <= circle - 1) ||
                            (c >= in_size + circle &&
                             c <= in_size + 2 * circle - 1)) {
                            out[row][col] = 0;
                        } else {
                            out[row][col] =
                                R.at<float>(r - paddings, c - paddings);
                        }
                    } else {
                        out[row][col] = R.at<float>(r, c);
                    }
                }
            }
            for (size_t r = i; r < i + kernel_size; r++) {
                for (size_t c = j; c < j + kernel_size; c++, row++) {
                    if (paddings > 0) {
                        if ((r >= 0 && r <= circle - 1) ||
                            (r >= in_size + circle &&
                             r <= in_size + 2 * circle - 1) ||
                            (c >= 0 && c <= circle - 1) ||
                            (c >= in_size + circle &&
                             c <= in_size + 2 * circle - 1)) {
                            out[row][col] = 0;
                        } else {
                            out[row][col] =
                                G.at<float>(r - paddings, c - paddings);
                        }
                    } else {
                        out[row][col] = G.at<float>(r, c);
                    }
                }
            }
            for (size_t r = i; r < i + kernel_size; r++) {
                for (size_t c = j; c < j + kernel_size; c++, row++) {
                    if (paddings > 0) {
                        if ((r >= 0 && r <= circle - 1) ||
                            (r >= in_size + circle &&
                             r <= in_size + 2 * circle - 1) ||
                            (c >= 0 && c <= circle - 1) ||
                            (c >= in_size + circle &&
                             c <= in_size + 2 * circle - 1)) {
                            out[row][col] = 0;
                        } else {
                            out[row][col] =
                                B.at<float>(r - paddings, c - paddings);
                        }
                    } else {
                        out[row][col] = B.at<float>(r, c);
                    }
                }
            }
        }
    }
}

void convResToMat(Matrix<float>& in,
                  Matrix<float>& out,
                  size_t kernel_size,
                  size_t paddings,
                  size_t stride) {
    assert(in.getData() != nullptr);
    delete[] out.getData();
    auto in_size = (size_t)sqrt(in.getCols());
    //由公式可以计算卷积后得到的矩阵有多少行(列)
    size_t size = (in_size + 2 * paddings - kernel_size) / stride + 1;
    out.setRows(kernel_size * kernel_size * in.getRows());
    out.setCols(size * size);
    out.getData() = new float[out.getRows() * out.getCols()];
    memset(out.getData(), 0, sizeof(float) * out.getRows() * out.getCols());
    //将输出的多通道矩阵展开成列向量，并根据padding的值来确定是否要补0，以及kernel_size的值决定补几圈0
    // circle来确定补几圈0
    size_t circle = (kernel_size - 1) / 2;
    size_t col = 0;
    for (size_t i = 0, cnt1 = 0; cnt1 < size; i += stride, cnt1++) {
        for (size_t j = 0, cnt2 = 0; cnt2 < size; j += stride, col++, cnt2++) {
            size_t row = 0;
            for (size_t channel = 0; channel < in.getRows(); channel++) {
                for (size_t r = i; r < i + kernel_size; r++) {
                    for (size_t c = j; c < j + kernel_size; c++, row++) {
                        if (paddings > 0) {
                            if ((r >= 0 && r <= circle - 1) ||
                                (r >= in_size + circle &&
                                 r <= in_size + 2 * circle - 1) ||
                                (c >= 0 && c <= circle - 1) ||
                                (c >= in_size + circle &&
                                 c <= in_size + 2 * circle - 1)) {
                                out[row][col] = 0;
                            } else {
                                out[row][col] =
                                    in[channel]
                                      [(r - paddings) * in_size + c - paddings];
                            }
                        } else {
                            out[row][col] = in[channel][r * in_size + c];
                        }
                    }
                }
            }
        }
    }
}

void convAndRelu(Matrix<float>& in, conv_param& conv_p, Matrix<float>& out) {
    in.mul_openblas(
        conv_p.p_weight, conv_p.p_bias, conv_p.out_channels,
        conv_p.in_channels * conv_p.kernel_size * conv_p.kernel_size, out,
        true);
}

void maxPooling(Matrix<float>& in, Matrix<float>& out) {
    delete[] out.getData();
    auto size = (size_t)sqrt(in.getCols());
    out.setRows(in.getRows());
    out.setCols(in.getCols() / 4);
    out.getData() = new float[out.getRows() * out.getCols()];
    memset(out.getData(), 0, sizeof(float) * out.getRows() * out.getCols());
    float* mat = out.getData();
    for (size_t channel = 0; channel < in.getRows(); channel++) {
        for (size_t row = 0; row < size; row += 2) {
            for (size_t col = 0; col < size; col += 2) {
                float a =
                    in.getData()[channel * in.getCols() + row * size + col];
                float b =
                    in.getData()[channel * in.getCols() + row * size + col + 1];
                float c = in.getData()[channel * in.getCols() + row * size +
                                       col + size];
                float d = in.getData()[channel * in.getCols() + row * size +
                                       col + size + 1];
                //取2*2方块中的最大值
                *mat = std::max(a, std::max(b, std::max(c, d)));
                mat++;
            }
        }
    }
}

void fullyConnected(Matrix<float>& in, fc_param& fc_p, Matrix<float>& out) {
    in.mul_openblas(fc_p.p_weight, fc_p.p_bias, fc_p.out_features,
                    fc_p.in_features, out, false);
}