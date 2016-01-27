package com.arbor.ml;

/**
 * Perceptron 感知机学习算法对偶形式
 *                      Ν
 * 学习模型 f(x) = sign( ∑   α  y  x * x + b)
 *                     j=1  j  j  j   i
 *                     梯度下降更新α，b
 * Author: arbor
 * Date: 16-1-26.
 */
public class PerceptronDual extends PerceptronAbstract {

    /** alpha = α   α(i)=n(i)*learningRate n(i)为某一点误分类次数 */
    double[] alpha;

    /** gram 矩阵 先将训练集实例点间内积计算出来存储于矩阵中 */
    double[][] gram;

    public PerceptronDual(double[][] data) {
        super(data);

        int N = data.length;
        this.alpha = new double[N]; //α(i)均初始化为0
        this.gram = new double[N][N];
        calGram(data);
    }

    public PerceptronDual(double bias, double learningRate, double[][] input) {
        super(bias, learningRate, input);

        int N = input.length;
        this.alpha = new double[N]; //α(i)均初始化为0
        this.gram = new double[N][N];
        calGram(input);
    }

    /**
     * 计算gram矩阵
     * @param data 训练集数据
     */
    private void calGram(double[][] data) {
        int N = data.length; //实例点个数

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                gram[i][j] = dotProduct(data[i], data[j]);
            }
        }
    }

    /**
     * 求向量内积
     * 注：此处x,y中最后一项为1/-1,表示正负实例点，求内积时不计入，故length-1
     * @param x 向量x
     * @param y 向量y
     * @return 内积
     */
    private double dotProduct(double[] x, double[] y) {
        double result = 0;

        for (int i = 0; i < x.length - 1 ; i++) {
            result += x[i] * y[i];
        }

        return result;
    }

    /**
     * 判断是否为误分类点
     *     N
     * y ( ∑   α  y  x * x + b) <= 0 为误分类
     *  i  j=1  j  j  j   i
     *
     * @param sampleIndex 实例点索引
     * @return true误分类 false否
     */
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

    /**
     * 根据随机梯度下降算法更新
     * α(i) <- α(i) + learningRate
     * b <- b + learningRate * y(i)
     * @param sampleIndex 实例点索引
     */
    @Override
    protected void gradientDescent(int sampleIndex) {
        double [] sample = input[sampleIndex];
        int yIndex = sample.length - 1;

        alpha[sampleIndex] += learningRate;
        bias += learningRate * sample[yIndex];
    }

    /**
     * 根据α 求得w
     * @return w权值向量
     */
    @Override
    public double[] getW() {
        int N = input.length;
        int yIndex = input[0].length - 1;
        int dim = yIndex;
        double[] w = new double[dim];

        for (int i = 0; i < N; i++) {
            double num = alpha[i] * input[i][yIndex];

            double[] item = numVectorProduct(input[i],num);
            w = vectorAdd(w, item);
        }
        return w;
    }

    /**
     * 向量相加
     * @param x 向量x
     * @param y 向量y
     * @return 与x同维的double[]
     */
    private double[] vectorAdd(double[] x, double[] y) {
        int dim = x.length;
        double[] result = new double[dim];

        for (int i = 0; i < dim; i++) {
            result[i] = x[i] + y[i];
        }

        return result;
    }

    /**
     * 数乘向量
     * @param x 实例点向量，包含最后一位标志正负实例点的 1/-1
     * @param y 数y
     * @return 比x少一维的result[]
     */
    private double[] numVectorProduct(double[] x, double y) {
        int dim = x.length - 1;
        double[] result = new double[dim];

        for (int i = 0; i < dim ; i++) {
            result[i] = y * x[i];
        }

        return result;
    }

    public double[] getAlpha() {
        return alpha;
    }
}
