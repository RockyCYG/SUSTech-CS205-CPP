#pragma once

#include <iostream>

template<typename T>
class Matrix {
private:
    size_t rows{};
    size_t cols{};
    T *data;
    int *refcount{};

public:
    Matrix();

    Matrix(size_t row, size_t col);

    Matrix(const Matrix<T> &m);

    ~Matrix();

    [[nodiscard]] size_t getRows() const;

    [[nodiscard]] size_t getCols() const;

    T *&getData();

    void setRows(size_t row);

    void setCols(size_t col);

    void release();

    void mul(T *weight, T *bias, size_t row, size_t col, Matrix<T> &out, bool flag) const;

    Matrix<T> operator+(const Matrix<T> &m) const;

    Matrix<T> operator-(const Matrix<T> &m) const;

    Matrix<T> operator*(const Matrix<T> &m) const;

    T &operator()(size_t row, size_t col) const;

    T *operator[](size_t row) const;

    template<typename E>
    friend std::ostream &operator<<(std::ostream &os, const Matrix<E> &m);

    template<typename E>
    friend std::istream &operator>>(std::istream &is, Matrix<E> &m);
};

template<typename T>
Matrix<T>::Matrix() : rows(0), cols(0), data(nullptr), refcount(nullptr) {}

template<typename T>
Matrix<T>::Matrix(size_t row, size_t col) : rows(row), cols(col) {
    data = new T[row * col];
    memset(data, 0, sizeof(T) * row * col);
    refcount = new int(0);
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T> &m) {
    if (m.refcount) {
        (*(m.refcount))++;
    }
    this->rows = m.rows;
    this->cols = m.cols;
    this->refcount = m.refcount;
    this->data = m.data;
}

template<typename T>
Matrix<T>::~Matrix() {
    release();
}

template<typename T>
size_t Matrix<T>::getRows() const {
    return rows;
}

template<typename T>
size_t Matrix<T>::getCols() const {
    return cols;
}

template<typename T>
void Matrix<T>::setRows(size_t row) {
    this->rows = row;
}

template<typename T>
void Matrix<T>::setCols(size_t col) {
    this->cols = col;
}

template<typename T>
T *&Matrix<T>::getData() {
    return data;
}

template<typename T>
void Matrix<T>::release() {
    if (refcount != nullptr) {
        if (*refcount == 0) {
            delete refcount;
            delete[] data;
        } else {
            (*refcount)--;
        }
    }
    this->rows = 0;
    this->cols = 0;
    this->data = nullptr;
    this->refcount = nullptr;
}

//实现的是weight * Matrix并且加上偏移量bias，flag变量来确定此次要不要做relu操作
template<typename T>
void Matrix<T>::mul(T *weight, T *bias, size_t row, size_t col, Matrix<T> &out, bool flag) const {
    assert(col == this->rows);
    if (out.data) {
        delete[] out.data;
    }
    out.rows = row;
    out.cols = this->cols;
    out.data = new T[row * this->cols];
    memset(out.data, 0, sizeof(T) * row * this->cols);
    for (size_t i = 0; i < row; i++) {
        for (size_t k = 0; k < col; k++) {
            T t = weight[i * col + k];
            for (size_t j = 0; j < this->cols; j++) {
                out.data[i * this->cols + j] += t * this->data[k * this->cols + j];
            }
        }
    }
    for (size_t channel = 0; channel < out.rows; channel++) {
        for (size_t y = 0; y < out.cols; y++) {
            out[channel][y] += bias[channel];
            if (flag) {
                out[channel][y] = std::max(0.0f, out[channel][y]);
            }
        }
    }
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &m) const {
    assert(this->rows == m.rows && this->cols == m.cols);
    Matrix add(this->rows, this->cols);
    size_t length = this->rows * this->cols;
    const T *p1 = this->data;
    const T *p2 = m.data;
    float *p3 = add.data;
    for (size_t i = 0; i < length; i++) {
        *(p3++) = *(p1++) + *(p2++);
    }
    return add;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &m) const {
    assert(this->rows == m.rows && this->cols == m.cols);
    Matrix sub(this->rows, this->cols);
    size_t length = this->rows * this->cols;
    const T *p1 = this->data;
    const T *p2 = m.data;
    float *p3 = sub.data;
    for (size_t i = 0; i < length; i++) {
        *(p3++) = *(p1++) - *(p2++);
    }
    return sub;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &m) const {
    assert(this->cols == m.rows);
    Matrix<T> mul(this->rows, m.cols);
    //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, this->rows, m.cols, this->cols, 1, this->data, this->cols, m.data, m.cols, 0, mul.data, m.cols);
    return mul;
}

template<typename T>
T &Matrix<T>::operator()(size_t row, size_t col) const {
    return data[row * this->cols + col];
}

template<typename T>
T *Matrix<T>::operator[](size_t row) const {
    return data + row * cols;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &m) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            os << m.data[i * m.cols + j] << " ";
        }
        os << std::endl;
    }
    return os;
}

template<typename T>
std::istream &operator>>(std::istream &is, Matrix<T> &m) {
    is >> m.rows;
    is >> m.cols;
    m.data = new T[m.rows * m.cols];
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            is >> m.data[i * m.cols + j];
        }
    }
    return is;
}