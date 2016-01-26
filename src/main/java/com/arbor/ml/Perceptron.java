
package com.arbor.ml;

import java.util.Arrays;

/**
 * Perceptron 感知机学习算法原始形式
 * 学习模型 f(x) = sign(w * x + b)
 */
public class Perceptron extends PerceptronAbstract {

    /** 权值向量weight, 维度根据训练数据x的维度设定 */
    private double[] w;

    /**
     * 感知机构造器
     * @param data 输入训练集，（x,y）, 最后一项为yi值
     *              eg. {4,3,1} 为x=(4,3)的正实例点
     *             还可加一个实例点类Point { {4,3}, 1}
     */
    public Perceptron(double[][] data) {
        super(data);

        int wDim = data[0].length - 1;
        this.w = new double[wDim];
    }

    public Perceptron(double[][] data, double learningRate, double bias) {
        super(bias, learningRate, data);

        int wDim = data[0].length - 1;
        this.w = new double[wDim];
    }

    /**
     * 学习算法，循环取训练集中数据判断误分类点，并更新w,b 直至没有误分类点
     * 须针对线性可分数据集，应有一个误分类次数上限，防止陷入死循环
     */
    public void train() {
        boolean error;
        boolean allCorrect = false;

        while (!allCorrect) {
            error = false;
            for (int i = 0; i < input.length ; i++) {
                if (hasError(input[i])) { // 存在误分类点，则以此点梯度下降更新w,b
                    gradientDescent(input[i]);
                    error = true;
                    break;
                }
            }
            if (!error)
                allCorrect = true;
        }

        System.out.println(Arrays.toString(w));
        System.out.println(bias);
    }

    /**
     * 判断是否有误分类点 yi * (w * xi + b) <=0 为误分类
     * @param sample 实例点
     * @return true有 false无
     */
    protected boolean hasError(double[] sample) {
        double temp = 0.0;
        int len = sample.length;

        for (int i = 0; i < len - 1 ; i++) {
            temp += w[i] * sample[i];
        }

        return (sample[len-1] * (temp + bias) <= 0);
    }

    /**
     * 根据随机梯度下降算法更新
     * @param sample 一个实例点
     */
    protected void gradientDescent(double[] sample) {
        int yIndex = sample.length - 1;

        for (int i = 0; i < w.length; i++) {
            w[i] += learningRate * sample[yIndex] * sample[i];
        }
        bias += learningRate * sample[yIndex];

        System.out.println("weight:" + Arrays.toString(w) + " bias:"+bias);
    }

    public double[] getW() {
        return w;
    }

    public double getBias() {
        return bias;
    }

    public static void main( String[] args )
    {
        //double[][] data = { {3, 3, 1}, {4, 3, 1}, {1, 1, -1} };
        double[][] data = {
                {-0.4, 0.3, 1}, {-0.3, -0.1, 1}, {-0.2, 0.4, 1},
                {-0.1, 0.1, 1}, {0.1, -0.5, -1}, {0.2, -0.9, -1},
                {0.3, 0.2, -1}, {0.4, -0.6, -1}
        };

        Perceptron per = new Perceptron(data, 1, 0);
        per.train();

        // 测试分类结果准确性
        double[] w = per.getW();
        double bias = per.getBias();

        for (int i = 0; i < data.length; i++) {
            System.out.println(w[0]*data[i][0] + w[1]*data[i][1] + bias);
        }
    }
}
