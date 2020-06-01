package hr.fer.zemris.zr.util;

import hr.fer.zemris.zr.data.HOGImage;

import java.awt.image.BufferedImage;
import java.util.concurrent.BlockingQueue;

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
     * Offset used in positive examples.
     */
    private static final int POSITIVE_OFFSET = 3;

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

            targetQueue.offer(positiveCalculationJob(image));
        }
    }

    /**
     * Job dont by {@link #positiveCalculationThread(BlockingQueue, BlockingQueue, BufferedImage)}.
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

        final float[] scaledMagnitude = new float[DEFAULT_HEIGHT * DEFAULT_WIDTH * pixelCount];
        final float[] scaledAngle = new float[DEFAULT_HEIGHT * DEFAULT_WIDTH * pixelCount];
        int index = 0;

        for (int i = POSITIVE_OFFSET, size = DEFAULT_HEIGHT + POSITIVE_OFFSET; i < size; i++) {
            int offset = i * width * pixelCount + POSITIVE_OFFSET * pixelCount;

            for (int j = 0; j < DEFAULT_WIDTH; j++) {
                for (int k = 0; k < pixelCount; k++) {
                    scaledMagnitude[index] = magnitude[offset];
                    scaledAngle[index++] = angle[offset++];
                }
            }
        }

        HOGImage hogImage = new HOGImage(scaledMagnitude, scaledAngle, DEFAULT_WIDTH, DEFAULT_HEIGHT, pixelCount);
        return hogImage.calculateWindow(0, false);
    }
}
