package hr.fer.zemris.zr.util;

import hr.fer.zemris.zr.data.HOGImage;

import java.awt.image.BufferedImage;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * A class that contains utility used by threads.
 *
 * @author Mateo Imbrišak
 */

public class ThreadsJobs {

    /**
     * Default width of the window used to calculate a feature vector.
     */
    private static final int DEFAULT_WIDTH = 64;

    /**
     * Default height of the window used to calculate a feature vector.
     */
    private static final int DEFAULT_HEIGHT = 128;

    /**
     * Step size used in {@link HOGImage#slideWindow(boolean, boolean)}.
     */
    private static final int STEP_SIZE = 5;

    /**
     * Number of samples per image used for {@link #negativeCalculationJob(BufferedImage, ThreadLocalRandom)}.
     */
    public static final int NUMBER_OF_SAMPLES = 10;

    /**
     * Don't let anyone instantiate this class.
     */
    private ThreadsJobs() {}

    /**
     * Thread body used to calculate vectors for positive examples.
     *
     * @param queue containing read images.
     * @param targetQueue used to store calculated vectors.
     * @param killElement used to stop the thread.
     */
    public static void positiveCalculationThread(BlockingQueue<BufferedImage> queue,
                                                 BlockingQueue<float[]> targetQueue, BufferedImage killElement) {
        while (true) {
            BufferedImage image;

            try {
                image = queue.take();
            } catch (InterruptedException exc) {
                continue;
            }

            if (image == killElement) {
                break;
            }

            if (image.getHeight() != DEFAULT_HEIGHT || image.getWidth() != DEFAULT_WIDTH) {
                continue;
            }

            targetQueue.offer(positiveCalculationJob(image));
        }
    }

    /**
     * Job done by {@link #positiveCalculationThread(BlockingQueue, BlockingQueue, BufferedImage)}.
     *
     * @param image currently being processed.
     *
     * @return feature vector of the given image.
     */
    private static float[] positiveCalculationJob(BufferedImage image) {
        final int width = image.getWidth();
        final int height = image.getHeight();

        final int pixelCount = image.getAlphaRaster() == null ? 3 : 4;

        final float[] data = Algorithms.getDataFloat(image);
        final float[] dx = Algorithms.derive(data, 'x', width, height, pixelCount, false);
        final float[] dy = Algorithms.derive(data, 'y', width, height, pixelCount, false);

        final float[] magnitude = Algorithms.calculateMagnitude(dx, dy);
        final float[] angle = Algorithms.calculateAngle(dx, dy);

        HOGImage hogImage = new HOGImage(magnitude, angle, DEFAULT_WIDTH, DEFAULT_HEIGHT, pixelCount);
        return hogImage.calculateWindow(0, false);
    }

    /**
     * Thread body used to calculate vectors for negative examples.
     *
     * @param queue containing read images.
     * @param targetQueue used to store calculated vectors.
     * @param killElement used to stop the thread.
     */
    public static void negativeCalculationThread(BlockingQueue<BufferedImage> queue,
                                                 BlockingQueue<float[]> targetQueue, BufferedImage killElement) {
        ThreadLocalRandom random = ThreadLocalRandom.current();
        while (true) {
            BufferedImage image;

            try {
                image = queue.take();
            } catch (InterruptedException exc) {
                continue;
            }

            if (image == killElement) {
                break;
            }

            List<float[]> images = negativeCalculationJob(image, random);

            for (float[] vector : images) {
                targetQueue.offer(vector);
            }
        }
    }

    /**
     * Job done by {@link #negativeCalculationThread(BlockingQueue, BlockingQueue, BufferedImage)}.
     *
     * @param image currently being processed.
     * @param random used to generate sample indexes.
     *
     * @return a {@link List} of randomly selected feature vectors of the given image.
     */
    private static List<float[]> negativeCalculationJob(BufferedImage image, ThreadLocalRandom random) {
        final HOGImage hogImage = new HOGImage(image);
        final int numberOfPositions = Algorithms.calculateNumberOfWindows(image.getWidth(), image.getHeight(),
                DEFAULT_WIDTH, DEFAULT_HEIGHT, STEP_SIZE);

        final List<float[]> vectors = new LinkedList<>();

        for (int i = 0; i < NUMBER_OF_SAMPLES; i++) {
            vectors.add(hogImage.calculateWindow(random.nextInt(numberOfPositions), false));
        }

        return vectors;
    }

    /**
     * Thread job used to calculate a feature vector.
     *
     * @param queue containing indexes of image parts to calculate.
     * @param featureVectors array used to store calculated feature vectors.
     * @param parallel used to toggle {@link HOGImage#calculateWindow(int, boolean)} parallelization.
     * @param function used to calculate feature vector for sliding window position drawn from {@code queue}.
     */
    public static void windowThread(BlockingQueue<Integer> queue, float[][] featureVectors, boolean parallel,
                                    BiFunction<Integer, Boolean, float[]> function) {
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

            featureVectors[index] = function.apply(index, parallel);
        }
    }

    /**
     * Thread job used to calculate histograms or normalize blocks.
     *
     * @param queue containing offset indexes.
     * @param functionParameter depends on the {@code function} used.
     * @param target array used to store normalized blocks.
     * @param function used to calculate histograms or normalize blocks.
     */
    public static <T> void hogThread(BlockingQueue<Integer> queue, T functionParameter, float[][] target,
                                       BiFunction<Integer, T, float[]> function) {
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

            target[currentIndex] = function.apply(currentIndex, functionParameter);
        }
    }

    /**
     * Thread job used to calculate histogram data.
     *
     * @param queue containing the indexes of histogram positions.
     * @param function used for calculation.
     * @param target array used to store calculated data.
     */
    public static void calculationThread(BlockingQueue<Integer> queue, Function<Integer, float[]> function,
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
}
