package com.arbor.ml;

import java.util.Arrays;

/**
 * Perceptron 感知机学习算法对偶形式
 *                      Ν
 * 学习模型 f(x) = sign( ∑   α  y  x * x + b)
 *                     j=1  j  j  j   i
 * Author: arbor
 * Date: 16-1-26.
 */
public class PerceptronDual extends PerceptronAbstract {

    /** alpha = α   α(i)=n(i)*learningRate n(i)为某一点误分类次数 */
    double[] alpha;

    double[][] gram;

    public PerceptronDual(double[][] data) {
        super(data);

        int N = data.length;
        this.alpha = new double[N]; //α(i)均初始化为0
        this.gram = new double[N][N];
        calGram(data);
    }

    private void calGram(double[][] data) {
        int N = data.length; //实例点个数

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                gram[i][j] = dotProduct(data[i], data[j]);
            }
        }
    }

    private double dotProduct(double[] x, double[] y) {
        double result = 0;

        for (int i = 0; i < x.length - 1 ; i++) {
            result += x[i] * y[i];
        }

        return result;
    }

    @Override
    public void train() {
        boolean error;
        boolean allCorrect = false;

        while (!allCorrect) {
            error = false;
            for (int i = 0; i < input.length ; i++) {
                if (hasError(i)) { // 存在误分类点，则以此点梯度下降更新w,b
                    gradientDescent(i);
                    error = true;
                    break;
                }
            }
            if (!error)
                allCorrect = true;
        }

        System.out.println(Arrays.toString(alpha));
        System.out.println(bias);
    }

    @Override
    protected boolean hasError(int sampleIndex) {
        double temp = 0.0;
        double[] sample = input[sampleIndex];
        int sampLen = sample.length;
        int N = input.length;

        for (int i = 0; i < N; i++) {
            temp += alpha[i] * input[i][sampLen-1] * gram[sampleIndex][i];
        }

        return (input[sampleIndex][sampLen-1] *  (temp + bias) <= 0);
    }

    @Override
    protected void gradientDescent(int sampleIndex) {
        double [] sample = input[sampleIndex];
        int yIndex = sample.length - 1;

        alpha[sampleIndex] += learningRate;
        bias += learningRate * sample[yIndex];
    }
}
