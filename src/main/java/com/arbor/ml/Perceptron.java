package com.arbor.ml;

import java.util.Arrays;

/**
 * Perceptron 感知机学习算法原始形式
 * 学习模型 f(x) = sign(w * x + b)
 * 梯度下降更新w,b
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
     * 判断是否有误分类点 yi * (w * xi + b) <=0 为误分类
     * @param sampleIndex 实例点索引
     * @return true有 false无
     */
    protected boolean hasError(int sampleIndex) {
        double temp = 0.0;
        double[] sample = input[sampleIndex];
        int len = sample.length;

        for (int i = 0; i < len - 1 ; i++) {
            temp += w[i] * sample[i];
        }

        return (sample[len-1] * (temp + bias) <= 0);
    }

    /**
     * 根据随机梯度下降算法更新
     * w <- w + learningRate * y(i) * x(i)  注x(i)为向量
     * b <- b + learningRate * y(i)
     * @param sampleIndex 一个实例点
     */
    protected void gradientDescent(int sampleIndex) {
        double[] sample = input[sampleIndex];
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

    public static void main( String[] args ) {
        double[][] data = { {3, 3, 1}, {4, 3, 1}, {1, 1, -1} };
        /*double[][] data = {
                {-0.4, 0.3, 1}, {-0.3, -0.1, 1}, {-0.2, 0.4, 1},
                {-0.1, 0.1, 1}, {0.1, -0.5, -1}, {0.2, -0.9, -1},
                {0.3, 0.2, -1}, {0.4, -0.6, -1}
        };*/

        PerceptronAbstract per = new PerceptronDual(data);
        //PerceptronDual per = new PerceptronDual(data);
        per.train();

        double[] w = per.getW();
        double bias = per.getBias();
        double[] alpha = ((PerceptronDual) per).getAlpha();

        System.out.println(Arrays.toString(w));
        System.out.println(bias);
        System.out.println(Arrays.toString(alpha));
        // 测试分类结果准确性
        /*double[] w = per.getW();
        double bias = per.getBias();

        for (int i = 0; i < data.length; i++) {
            System.out.println(w[0]*data[i][0] + w[1]*data[i][1] + bias);
        }*/
    }
}
