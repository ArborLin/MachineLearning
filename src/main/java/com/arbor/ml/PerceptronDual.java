package com.arbor.ml;

/**
 * Author: arbor
 * Date: 16-1-26.
 */
public class PerceptronDual extends PerceptronAbstract {

    public PerceptronDual(double[][] data) {
        super(data);

    }

    @Override
    public void train() {

    }

    @Override
    protected boolean hasError(double[] sample) {
        return false;
    }

    @Override
    protected void gradientDescent(double[] sample) {

    }
}
