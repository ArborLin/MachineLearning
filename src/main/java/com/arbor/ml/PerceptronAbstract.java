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

    /**
     * 学习算法，循环取训练集中数据判断误分类点，利用梯度下降更新 直至没有误分类点
     * 须针对线性可分数据集，应有一个误分类次数上限，防止陷入死循环
     */
    public void train() {
        boolean error;
        boolean allCorrect = false;

        while (!allCorrect) {
            error = false;
            for (int i = 0; i < input.length ; i++) {
                if (hasError(i)) { // 存在误分类点，则以此点梯度下降更新
                    gradientDescent(i);
                    error = true;
                    break;
                }
            }
            if (!error)
                allCorrect = true;
        }
    }

    /**
     * 判断是否为误分类点
     * @param sampleIndex 实例点索引
     * @return true为误分类 false为正确分类
     */
    protected abstract boolean hasError(int sampleIndex);

    /**
     * 根据随机梯度下降算法更新
     * @param sampleIndex 实例点索引
     */
    protected abstract void gradientDescent(int sampleIndex);

    public double getBias() {
        return bias;
    }

    public abstract double[] getW();
}
