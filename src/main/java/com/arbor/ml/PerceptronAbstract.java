package com.arbor.ml;

/**
 * PerceptronAbstract 感知机抽象类
 * Author: arbor
 * Date: 16-1-26.
 */
public abstract class PerceptronAbstract {
    /**  偏置值bias */
    protected double bias;

    /** 学习率，用于梯度下降算法，值在0-1间 */
    protected double learningRate;

    /** 输入训练集 */
    protected double[][] input;

    public PerceptronAbstract(double[][] input) {
        this.bias = 0;
        this.learningRate = 1;
        this.input = input;
    }

    public PerceptronAbstract(double bias, double learningRate, double[][] input) {
        this.bias = bias;
        this.learningRate = learningRate;
        this.input = input;
    }

    public abstract void train();

    protected abstract boolean hasError(int sampleIndex);

    protected abstract void gradientDescent(int sampleIndex);
}
