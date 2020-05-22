package hr.fer.zemris.zr.data;

import hr.fer.zemris.zr.util.Algorithms;

import java.awt.image.BufferedImage;
import java.util.concurrent.BlockingQueue;
import java.util.function.Function;

/**
 * A class that keeps data about histogram of gradients of the given image.
 *
 * @author Mateo Imbrišak
 */

public class HOGImage {

    /**
     * Required width of image to be used with the algorithm.
     */
    private static final int DEFAULT_WIDTH = 64;

    /**
     * Required height of image to be used with the algorithm.
     */
    private static final int DEFAULT_HEIGHT = 128;

    /**
     * Size of the window used.
     */
    private static final int WINDOW_SIZE = 8;

    /**
     * Number of bins used in histograms.
     */
    private static final int BIN_SIZE = 9;

    /**
     * Width of a bin in angles.
     */
    private static final int BIN_WIDTH = 20;

    /**
     * Block size used in normalization.
     */
    private static final int BLOCK_SIZE = 2;

    /**
     * Small positive value used to prevent division by zero.
     */
    private static final float EPSILON = 1E-5f;

    /**
     * Used as a maximum value while calculating {@link #featureVector}.
     */
    private static final float TAU = 0.2f;

    /**
     * Step size used in {@link #slideWindow(boolean, boolean)}.
     */
    private static final int STEP_SIZE = 5;

    /**
     * Keeps the calculated histogram data.
     */
    private float[][] histogram;

    /**
     * Keeps normalized blocks calculated by {@link #normalizeBlocks()}
     */
    private float[][] normalized;

    /**
     * Keeps calculated data as a feature vector used in detection.
     */
    private double[] featureVector;

    /**
     * Keeps the calculated magnitude of the image.
     */
    private final float[] magnitude;

    /**
     * Keeps calculated angle of the image.
     */
    private final float[] angle;

    /**
     * Keeps the width of the image.
     */
    private final int width;

    /**
     * Keeps the height of the image.
     */
    private final int height;

    /**
     * Keeps the number of bytes per pixel.
     */
    private final int bytesPerPixel;

    /**
     * Constructor that assigns all values.
     *
     * @param magnitude calculated magnitude of an image.
     * @param angle calculated angle of an image.
     * @param width of the image.
     * @param height of the image.
     * @param bytesPerPixel in the original image.
     */
    public HOGImage(float[] magnitude, float[] angle, int width, int height, int bytesPerPixel) {
        this.magnitude = magnitude;
        this.angle = angle;
        this.width = width;
        this.height = height;
        this.bytesPerPixel = bytesPerPixel;
    }

    /**
     * Constructor that calculates all values from the image.
     *
     * @param bim image to be represented.
     */
    public HOGImage(BufferedImage bim) {
        width = bim.getWidth();
        height = bim.getHeight();
        bytesPerPixel = bim.getAlphaRaster() == null ? 3 : 4;

        final float[] data = Algorithms.getDataFloat(bim);
        final float[] gx = Algorithms.derive(data, 'x', width, height, bytesPerPixel, true);
        final float[] gy = Algorithms.derive(data, 'y', width, height, bytesPerPixel, true);

        magnitude = Algorithms.calculateMagnitude(gx, gy);
        angle = Algorithms.calculateAngle(gx, gy);
    }

    /**
     * Calculates the histogram data.
     *
     * @return histogram data {@link #histogram}.
     *
     * @throws UnsupportedOperationException if the image is not {@code 64 x 128}.
     */
    public float[][] calculateHistogram() {
        if (histogram != null) {
            return histogram;
        }

        if (width != DEFAULT_WIDTH && height != DEFAULT_HEIGHT) {
            throw new UnsupportedOperationException("Width must be " + DEFAULT_WIDTH + " and height must be "
                    + DEFAULT_HEIGHT + "!");
        }

        final int totalSize = DEFAULT_HEIGHT / WINDOW_SIZE * DEFAULT_WIDTH / WINDOW_SIZE;
        histogram = new float[totalSize][BIN_SIZE];

        final int processors = Runtime.getRuntime().availableProcessors();

        BlockingQueue<Integer> queue = Algorithms.initializeQueue(totalSize, processors, -1);

        Thread[] threads = new Thread[processors];

        for (int i = 0; i < processors; i++) {
            Thread thread = new Thread(() -> calculationThread(queue, this::calculateHistogram, this.histogram));
            threads[i] = thread;

            thread.start();
        }

        Algorithms.joinThreads(threads);

        return histogram;
    }

    /**
     * Normalizes calculated {@link #histogram} as blocks of size {@link #BLOCK_SIZE} by {@link #BLOCK_SIZE}.
     *
     * @return array of normalized blocks.
     */
    public float[][] normalizeBlocks() {
        if (normalized != null) {
            return normalized;
        }

        if (histogram == null) {
            calculateHistogram();
        }

        final int rowSize = DEFAULT_WIDTH / WINDOW_SIZE - 1;
        final int columnSize = DEFAULT_HEIGHT / WINDOW_SIZE - 1;
        final int totalSize = rowSize * columnSize;

        normalized = new float[totalSize][BIN_SIZE * BLOCK_SIZE * BLOCK_SIZE];

        final int processors = Runtime.getRuntime().availableProcessors();

        BlockingQueue<Integer> queue = Algorithms.initializeQueue(totalSize, processors, -1);

        Thread[] threads = new Thread[processors];

        for (int i = 0; i < processors; i++) {
            Thread thread = new Thread(() -> calculationThread(queue, this::normalize, this.normalized));
            threads[i] = thread;

            thread.start();
        }

        Algorithms.joinThreads(threads);

        return normalized;
    }

    /**
     * Calculates the feature vector from this image.
     *
     * @return calculated feature vector.
     */
    public double[] calculateFeatureVector() {
        if (featureVector != null) {
            return featureVector;
        }

        if (normalized == null) {
            normalizeBlocks();
        }

        featureVector = calculateFeatureVector(normalized);
        return featureVector;
    }

    /**
     * Calculates the feature vector from a part of this image.
     *
     * @param normalized array containing normalized block values.
     *
     * @return calculated feature vector.
     */
    private double[] calculateFeatureVector(float[][] normalized) {
        double[] featureVector = new double[normalized.length * normalized[0].length];

        int index = 0;
        double sum = 0;

        for (float[] block : normalized) {
            for (float value : block) {
                sum += value * value;
                featureVector[index++] = value;
            }
        }

        sum = Math.sqrt(sum + EPSILON);
        index = 0;
        double normalizedSum = 0;

        for (double value : featureVector) {
            value = Math.min(value / sum, TAU);
            normalizedSum += value * value;
            featureVector[index++] = value;
        }

        normalizedSum = Math.sqrt(normalizedSum + EPSILON);

        for (int i = 0, size = featureVector.length; i < size; i++) {
            featureVector[i] /= normalizedSum;
        }

        return featureVector;
    }

    /**
     * Simulates sliding window mechanism and calculates feature vector
     * for every position.
     *
     * @param parallel whether feature vector calculation is parallel.
     * @param parallelChildren whether histogram calculation and normalization is parallel.
     *
     * @return an array containing calculated feature vector.
     */
    public double[][] slideWindow(boolean parallel, boolean parallelChildren) {
        final int processors = Runtime.getRuntime().availableProcessors();
        BlockingQueue<Integer> queue = Algorithms.initializeQueue(width, height, DEFAULT_WIDTH, DEFAULT_HEIGHT,
                STEP_SIZE, processors, -1);

        final int totalSize = queue.size() - processors;
        final int normalizedSize = (DEFAULT_WIDTH / WINDOW_SIZE - 1) * (DEFAULT_HEIGHT / WINDOW_SIZE - 1);

        final double[][] featureVectors = new double[totalSize][normalizedSize * BIN_SIZE * BLOCK_SIZE * BLOCK_SIZE];

        if (parallel) {
            Thread[] threads = new Thread[processors];

            for (int i = 0; i < processors; i++) {
                Thread thread = new Thread(() -> windowThread(queue, featureVectors, parallelChildren));
                threads[i] = thread;

                thread.start();
            }

            Algorithms.joinThreads(threads);
        } else {
            for (int i = 0; i < totalSize; i++) {
                featureVectors[i] = calculateWindow(i, parallelChildren);
            }
        }

        return featureVectors;
    }

    /**
     * Calculates feature vector in the given window.
     *
     * @param index used to indicate window position.
     * @param parallel whether the calculation is parallel or not.
     *
     * @return calculated feature vector.
     */
    private double[] calculateWindow(int index, boolean parallel) {
        final int histogramSize = DEFAULT_HEIGHT / WINDOW_SIZE * DEFAULT_WIDTH / WINDOW_SIZE;
        final int normalizedSize = (DEFAULT_WIDTH / WINDOW_SIZE - 1) * (DEFAULT_HEIGHT / WINDOW_SIZE - 1);

        final float[][] histogram = new float[histogramSize][BIN_SIZE];
        final float[][] normalized = new float[normalizedSize][BIN_SIZE * BLOCK_SIZE * BLOCK_SIZE];

        if (parallel) {
            final int processors = Runtime.getRuntime().availableProcessors();

            BlockingQueue<Integer> histogramQueue = Algorithms.initializeQueue(histogramSize, processors, -1);
            Thread[] threads = new Thread[processors];

            for (int i = 0; i < processors; i++) {
                Thread thread = new Thread(() -> histogramThread(histogramQueue, index, histogram));
                threads[i] = thread;

                thread.start();
            }

            Algorithms.joinThreads(threads);

            BlockingQueue<Integer> normalizationQueue =
                    Algorithms.initializeQueue(normalizedSize, processors, -1);

            for (int i = 0; i < processors; i++) {
                Thread thread = new Thread(() -> normalizationThread(normalizationQueue, normalized, histogram));
                threads[i] = thread;

                thread.start();
            }

            Algorithms.joinThreads(threads);
        } else {
            for (int i = 0; i < histogramSize; i++) {
                histogram[i] = calculateHistogram(i, index);
            }

            for (int i = 0; i < normalizedSize; i++) {
                normalized[i] = normalize(i, histogram);
            }
        }

        return calculateFeatureVector(normalized);
    }

    /**
     * Thread job used to calculate a feature vector.
     *
     * @param queue containing indexes of image parts to calculate.
     * @param featureVectors array used to store calculated feature vectors.
     * @param parallel used to toggle {@link #calculateWindow(int, boolean)} parallelization.
     */
    private void windowThread(BlockingQueue<Integer> queue, double[][] featureVectors, boolean parallel) {
        while (true) {
            int index;

            try {
                index = queue.take();
            } catch (InterruptedException exc) {
                continue;
            }

            if (index == -1) {
                break;
            }

            featureVectors[index] = calculateWindow(index, parallel);
        }
    }

    /**
     * Thread job used to calculate histograms.
     *
     * @param queue containing offset indexes.
     * @param index defines index of the image whose histogram is being calculated.
     * @param target array used to store normalized blocks.
     */
    private void histogramThread(BlockingQueue<Integer> queue, int index, float[][] target) {
        while (true) {
            int currentIndex;

            try {
                currentIndex = queue.take();
            } catch (InterruptedException exc) {
                continue;
            }

            if (currentIndex == -1) {
                break;
            }

            target[currentIndex] = calculateHistogram(currentIndex, index);
        }
    }

    /**
     * Thread job used to normalize blocks.
     *
     * @param queue containing block indexes.
     * @param target array used to store normalized blocks.
     * @param source array containing calculated histograms.
     */
    private void normalizationThread(BlockingQueue<Integer> queue, float[][] target, float[][] source) {
        while (true) {
            int currentIndex;

            try {
                currentIndex = queue.take();
            } catch (InterruptedException exc) {
                continue;
            }

            if (currentIndex == -1) {
                break;
            }

            target[currentIndex] = normalize(currentIndex, source);
        }
    }

    /**
     * Thread job used to calculate histogram data.
     *
     * @param queue containing the indexes of histogram positions.
     * @param function used for calculation.
     * @param target array used to store calculated data.
     */
    private void calculationThread(BlockingQueue<Integer> queue, Function<Integer, float[]> function,
                                   float[][] target) {
        while (true) {
            int index;

            try {
                index = queue.take();
            } catch (InterruptedException exc) {
                continue;
            }

            if (index == -1) {
                break;
            }

            target[index] = function.apply(index);
        }
    }

    /**
     * Calculates the histogram at the given index.
     *
     * @param position index of the histogram to be calculated.
     */
    private float[] calculateHistogram(int position) {
        return calculateHistogram(position, 0);
    }

    /**
     * Calculates the histogram at the given index.
     *
     * @param position index of the histogram to be calculated.
     * @param offsetIndex index of offset in the whole image.
     */
    private float[] calculateHistogram(int position, int offsetIndex) {
        final int windowRow = width / STEP_SIZE;
        final int rowIndex = offsetIndex / windowRow;
        final int columnOffset = offsetIndex % windowRow;
        final int windowOffset = rowIndex * width * bytesPerPixel + columnOffset * STEP_SIZE;

        final int widthSize = DEFAULT_WIDTH / WINDOW_SIZE;
        final int rowPosition = position / widthSize;
        final int rowOffset = (position - (WINDOW_SIZE * rowPosition)) * WINDOW_SIZE * bytesPerPixel;
        final int initialOffset = rowPosition * DEFAULT_WIDTH * WINDOW_SIZE * bytesPerPixel + rowOffset + windowOffset;

        float[] histogram = new float[BIN_SIZE];

        for (int i = 0; i < WINDOW_SIZE; i++) {
            int offset = initialOffset + i * width * bytesPerPixel;
            for (int j = 0; j < WINDOW_SIZE * bytesPerPixel; j++) {
                final float angle = this.angle[offset];
                final float magnitude = this.magnitude[offset];

                final int index = (int) (angle / BIN_WIDTH);
                final float factor = (angle - (BIN_WIDTH * index)) / BIN_WIDTH;

                histogram[index] += (1 - factor) * magnitude;
                histogram[(index + 1) % BIN_SIZE] += factor * magnitude;

                offset++;
            }
        }

        return histogram;
    }

    /**
     * Job used to normalize a calculated {@link #histogram} by blocks.
     *
     * @param position index of the block to be normalized.
     */
    private float[] normalize(int position) {
        return normalize(position, this.histogram);
    }

    /**
     * Job used to normalize a calculated {@link #histogram} by blocks.
     *
     * @param position index of the block to be normalized.
     * @param sourceHistogram histogram being normalized.
     */
    private float[] normalize(int position, float[][] sourceHistogram) {
        final int rowSize = DEFAULT_WIDTH / WINDOW_SIZE - 1;
        final int skipSize = DEFAULT_WIDTH / WINDOW_SIZE;
        final int row = position / rowSize;
        final int offset = row * (DEFAULT_WIDTH / WINDOW_SIZE) + (position % rowSize);

        float[] normalized = new float[BIN_SIZE * BLOCK_SIZE * BLOCK_SIZE];
        double sum = 0;
        int index = 0;

        for (int i = 0; i < BLOCK_SIZE; i++) {
            int currentOffset = offset + i * skipSize;
            for (int j = 0; j < BLOCK_SIZE; j++) {
                float[] histogram = sourceHistogram[currentOffset++];

                for (int k = 0; k < BIN_SIZE; k++) {
                    float value = histogram[k];
                    normalized[index * BIN_SIZE + k] = value;
                    sum += value * value;
                }

                index++;
            }
        }

        sum = Math.sqrt(sum + EPSILON);

        for (int i = 0, size = normalized.length; i < size; i++) {
            normalized[i] /= sum;
        }

        return normalized;
    }
}
