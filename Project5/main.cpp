#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include "cnn.cpp"

using namespace std;
using namespace cv;

//获取时间戳
time_t getTimeStamp()
{
    chrono::time_point<chrono::system_clock, chrono::milliseconds> tp = chrono::time_point_cast<chrono::milliseconds>(
            chrono::system_clock::now());
    time_t timestamp = tp.time_since_epoch().count();
    return timestamp;
}

//对传入的图像做完整的卷积操作，并返回人脸的概率
vector<float> CNN(const string &path)
{
    time_t start, end;
    cout << "读入图像：";
    start = getTimeStamp();
    Mat img = imread(path);
    end = getTimeStamp();
    cout << (end - start) << " ms" << endl;
    cout << "将图像中的数据转换成float类型：";
    start = getTimeStamp();
    img.convertTo(img, CV_32FC3, 1.0 / 255);
    end = getTimeStamp();
    cout << (end - start) << " ms" << endl;
    Matrix<float> input1, Relu1, MaxPooling1, input2, Relu2, MaxPooling2, input3, Relu3, finalResult;
    cout << "-----------------------------" << endl;
    cout << "第一次卷积：" << endl;
    cout << "imgToMat(): ";
    start = getTimeStamp();
    imgToMat(img, input1, conv_params[0].kernel_size, conv_params[0].pad, conv_params[0].stride);
    end = getTimeStamp();
    cout << (end - start) << " ms" << endl;
    cout << "convAndRelu(): ";
    start = getTimeStamp();
    convAndRelu(input1, conv_params[0], Relu1);
    end = getTimeStamp();
    cout << (end - start) << " ms" << endl;
    cout << "maxPooling(): ";
    start = getTimeStamp();
    maxPooling(Relu1, MaxPooling1);
    end = getTimeStamp();
    cout << (end - start) << "ms" << endl;
    cout << "convResToMat(): ";
    start = getTimeStamp();
    convResToMat(MaxPooling1, input2, conv_params[1].kernel_size, conv_params[1].pad, conv_params[1].stride);
    end = getTimeStamp();
    cout << (end - start) << "ms" << endl;
    cout << "-----------------------------" << endl;
    cout << "第二次卷积：" << endl;
    cout << "convAndRelu(): ";
    start = getTimeStamp();
    convAndRelu(input2, conv_params[1], Relu2);
    end = getTimeStamp();
    cout << (end - start) << " ms" << endl;
    cout << "maxPooling(): ";
    start = getTimeStamp();
    maxPooling(Relu2, MaxPooling2);
    end = getTimeStamp();
    cout << (end - start) << "ms" << endl;
    cout << "convResToMat(): ";
    start = getTimeStamp();
    convResToMat(MaxPooling2, input3, conv_params[2].kernel_size, conv_params[2].pad, conv_params[2].stride);
    end = getTimeStamp();
    cout << (end - start) << "ms" << endl;
    cout << "-----------------------------" << endl;
    cout << "第三次卷积：" << endl;
    cout << "convAndRelu(): ";
    start = getTimeStamp();
    convAndRelu(input3, conv_params[2], Relu3);
    end = getTimeStamp();
    cout << (end - start) << " ms" << endl;
    Relu3.setRows(2048);
    Relu3.setCols(1);
    cout << "fullyConnected(): ";
    start = getTimeStamp();
    fullyConnected(Relu3, fc_params[0], finalResult);
    end = getTimeStamp();
    cout << (end - start) << " ms" << endl;
    cout << "softMax: ";
    start = getTimeStamp();
    float x = finalResult.getData()[0];
    float y = finalResult.getData()[1];
    float sum = exp(x) + exp(y);
    float p1 = exp(x) / sum;
    float p2 = exp(y) / sum;
    end = getTimeStamp();
    cout << (end - start) << " ms" << endl;
    return {p1, p2};
}

int main()
{
    string path = "../samples/HuaShan.jpg";
    string picture = path.substr(11);
    time_t start = getTimeStamp();
    vector<float> ans = CNN(path);
    time_t end = getTimeStamp();
    cout << "-----------------------------" << endl;
    cout << "整个卷积流程时间：" << (end - start) << " ms" << endl;
    cout << endl;
    cout << picture << ": " << endl;
    cout << "bg score: " << ans[0] << endl;
    cout << "face score: " << ans[1] << endl;
    return 0;
}
