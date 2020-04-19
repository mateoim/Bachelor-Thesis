package hr.fer.zemris.zr.data;

import hr.fer.zemris.zr.util.Algorithms;

import java.awt.image.BufferedImage;
import java.util.concurrent.BlockingQueue;
import java.util.function.Consumer;

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
     * Keeps the calculated histogram data.
     */
    private float[][] histogram;

    /**
     * Keeps normalized blocks calculated by {@link #normalizeBlocks()}
     */
    private float[][] normalized;

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
            Thread thread = new Thread(() -> calculationThread(queue, this::calculateHistogram));
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

        final int rowSize = width / WINDOW_SIZE - 1;
        final int columnSize = height / WINDOW_SIZE - 1;
        final int totalSize = rowSize * columnSize;

        normalized = new float[totalSize][BIN_SIZE * BLOCK_SIZE * BLOCK_SIZE];

        final int processors = Runtime.getRuntime().availableProcessors();

        BlockingQueue<Integer> queue = Algorithms.initializeQueue(totalSize, processors, -1);

        Thread[] threads = new Thread[processors];

        for (int i = 0; i < processors; i++) {
            Thread thread = new Thread(() -> calculationThread(queue, this::normalize));
            threads[i] = thread;

            thread.start();
        }

        Algorithms.joinThreads(threads);

        return normalized;
    }

    /**
     * Thread job used to calculate histogram data.
     *
     * @param queue containing the indexes of histogram positions.
     * @param function used for calculation.
     */
    private void calculationThread(BlockingQueue<Integer> queue, Consumer<Integer> function) {
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

            function.accept(index);
        }
    }

    /**
     * Calculates the histogram at the given index.
     *
     * @param position index of the histogram to be calculated.
     */
    private void calculateHistogram(int position) {
        final int widthSize = DEFAULT_WIDTH / WINDOW_SIZE;
        final int rowPosition = position / widthSize;
        final int rowOffset = (position - (WINDOW_SIZE * rowPosition)) * WINDOW_SIZE * bytesPerPixel;
        final int initialOffset = rowPosition * DEFAULT_WIDTH * WINDOW_SIZE * bytesPerPixel + rowOffset;

        float[] histogram = this.histogram[position];

        for (int i = 0; i < WINDOW_SIZE; i++) {
            int offset = initialOffset + i * DEFAULT_WIDTH * bytesPerPixel;
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
    }

    /**
     * Job used to normalize a calculated {@link #histogram} by blocks.
     *
     * @param position index of the block to be normalized.
     */
    private void normalize(int position) {
        final int rowSize = width / WINDOW_SIZE - 1;
        final int skipSize = width / WINDOW_SIZE;
        final int row = position / rowSize;
        final int offset = row * (width / WINDOW_SIZE) + (position % rowSize);

        float[] normalized = this.normalized[position];
        double sum = 0;
        int index = 0;

        for (int i = 0; i < BLOCK_SIZE; i++) {
            int currentOffset = offset + i * skipSize;
            for (int j = 0; j < BLOCK_SIZE; j++) {
                float[] histogram = this.histogram[currentOffset++];

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
    }
}
