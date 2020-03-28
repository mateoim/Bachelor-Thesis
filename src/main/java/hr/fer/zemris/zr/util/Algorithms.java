package hr.fer.zemris.zr.util;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.function.Consumer;

/**
 * A class containing algorithms used in image
 * processing and calculations.
 *
 * @author Mateo Imbrišak
 */

public class Algorithms {

    private static final int MAX_ALPHA_INT = -16777216;

    /**
     * Don't let anyone instantiate this class.
     */
    private Algorithms() {}

    /**
     * Converts a {@link BufferedImage} to a {@code float} type {@code array}.
     * Size of the array is equal to width * height * number of bytes per pixel
     * (3 if image has on alpha channel, otherwise 4). Colors are ordered the
     * same way as they are in image's {@link DataBufferByte#getData()}.
     *
     * @param img image whose data is being processed.
     *
     * @return an array of {@code float} values that represent each pixel's color.
     */
    public static float[] getDataFloat(BufferedImage img) {
        final byte[] pixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        final int width = img.getWidth();
        final int height = img.getHeight();
        final boolean hasAlpha = img.getAlphaRaster() != null;
        final int valuesPerPixel = hasAlpha ? 4 : 3;

        final int size = width * height * valuesPerPixel;
        float[] data = new float[size];

        for (int i = 0; i < size; i += valuesPerPixel) {
            for (int j = 0; j < valuesPerPixel; j++) {
                data[i + j] = (pixels[i + j] & 0xFF) / 255f;
            }
        }

        return data;
    }

    /**
     * Converts a {@link BufferedImage} to a {@code int} type {@code array}.
     * Size of the array is equal to width * height. Colors are ordered the
     * same way as they are in image's {@link DataBufferByte#getData()}.
     *
     * @param img image whose data is being processed.
     *
     * @return an array of {@code int} values that represent each pixel's color.
     */
    public static int[] getDataInt(BufferedImage img) {
        final byte[] pixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        final int width = img.getWidth();
        final int height = img.getHeight();
        final boolean hasAlpha = img.getAlphaRaster() != null;

        final int size = width * height;
        int[] data = new int[size];

        int pixelCounter = 0;

        if (hasAlpha) {
            for (int i = 0; i < size; i++) {
                int rgb = 0;
                rgb += (((int) pixels[pixelCounter++] & 0xFF) << 24); // alpha
                rgb += ((int) pixels[pixelCounter++] & 0xFF); // blue
                rgb += (((int) pixels[pixelCounter++] & 0xFF) << 8); // green
                rgb += (((int) pixels[pixelCounter++] & 0xFF) << 16); // red

                data[i] = rgb;
            }
        } else {
            for (int i = 0; i < size; i++) {
                int rgb = MAX_ALPHA_INT;
                rgb += ((int) pixels[pixelCounter++] & 0xFF); // blue
                rgb += (((int) pixels[pixelCounter++] & 0xFF) << 8); // green
                rgb += (((int) pixels[pixelCounter++] & 0xFF) << 16); // red

                data[i] = rgb;
            }
        }

        return data;
    }

    /**
     * Calculates derivative of the image whose data is passed as {@code src}
     * in the direction defined by {@code direction}.
     *
     * @param src image data generated by {@link #getDataFloat(BufferedImage)}.
     * @param direction {@code x} or {@code y}.
     * @param width of the image.
     * @param height of the image.
     * @param pixelSize number of elements used to represent a pixel.
     * @param parallel whether the execution should be parallel or not.
     *
     * @return an {@code array} of {@code float} elements representing derivative
     *         in the given direction.
     */
    public static float[] derive(float[] src, char direction, int width, int height, int pixelSize, boolean parallel) {
        final int size = src.length;
        float[] dest = new float[size];

        if (parallel) {
            int processors = Runtime.getRuntime().availableProcessors();
            BlockingQueue<Integer> queue;
            Thread[] threads = new Thread[processors];
            int killElement = -1;

            switch (direction) {
                case 'x':
                    queue = initializeQueue(height, processors, killElement);
                    Consumer<Integer> derivativeX = integer -> deriveX(src, dest, width, pixelSize, integer);

                    for (int i = 0; i < processors; i++) {
                        threads[i] = new Thread(() -> deriveThread(queue, killElement, derivativeX));

                        threads[i].start();
                    }

                    joinThreads(threads);
                    break;
                case 'y':
                    queue = initializeQueue(width, processors, killElement);
                    Consumer<Integer> derivativeY = integer -> deriveY(src, dest, width, height, pixelSize, integer);

                    for (int i = 0; i < processors; i++) {
                        threads[i] = new Thread(() -> deriveThread(queue, killElement, derivativeY));

                        threads[i].start();
                    }

                    joinThreads(threads);
                    break;
                default:
                    throw new RuntimeException("Direction must be x or y.");
            }
        } else {
            switch (direction) {
                case 'x':
                    for (int i = 0; i < height; i++) {
                        deriveX(src, dest, width, pixelSize, i);
                    }
                    break;
                case 'y':
                    for (int i = 0; i < width; i++) {
                        deriveY(src, dest, width, height, pixelSize, i);
                    }
                    break;
                default:
                    throw new RuntimeException("Direction must be x or y.");
            }
        }

        return dest;
    }

    /**
     * Calculates the magnitude using given x and y derivatives.
     *
     * @param gx array containing x derivative.
     * @param gy array containing y derivative.
     *
     * @return an array containing calculated magnitude.
     */
    public static float[] calculateMagnitude(float[] gx, float[] gy) {
        if (gx.length != gy.length) {
            throw new RuntimeException("Both arrays must be of same length.");
        }

        int length = gx.length;
        float[] magnitude = new float[length];

        for (int i = 0; i < length; i++) {
            float x = gx[i];
            float y = gy[i];

            magnitude[i] = (float) Math.sqrt(x * x + y * y);
        }

        return magnitude;
    }

    /**
     * Creates a {@link BufferedImage} from the given {@code data} array.
     *
     * @param data array containing pixel data.
     * @param width of the image.
     * @param height of the image.
     * @param colorType defines {@code imageType} of created {@link BufferedImage}.
     *
     * @return a new {@link BufferedImage} whose pixels are calculated from {@code data}.
     */
    public static BufferedImage convertToImage(float[] data, int width, int height, int colorType) {
        BufferedImage img = new BufferedImage(width, height, colorType);
        byte[] pixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();

        int offset = 0;

        for (float color : data) {
            pixels[offset++] = (byte) (Math.round(Math.abs(color) * 255) & 0xFF);
        }

        return img;
    }

    /**
     * Used in methods that use multithreading. Causes main thread
     * to {@link Thread#join()} all threads in {@code threads}.
     *
     * @param threads that the main thread should join.
     */
    private static void joinThreads(Thread[] threads) {
        for (int i = 0, processors = threads.length; i < processors; i++) {
            try {
                threads[i].join();
            } catch (InterruptedException exc) {
                i--;
            }
        }
    }

    /**
     * Method that defines job done by {@link Thread}s in {@link #derive(float[], char, int, int, int, boolean)} method.
     *
     * @param queue containing indexes to derive.
     * @param killElement element used to terminate the thread.
     * @param derivative {@link Consumer} that contains derivative function.
     */
    private static void deriveThread(BlockingQueue<Integer> queue, int killElement, Consumer<Integer> derivative) {
        while (true) {
            int index;

            try {
                index = queue.take();
            } catch (InterruptedException exc) {
                continue;
            }

            if (index == killElement) {
                break;
            }

            derivative.accept(index);
        }
    }

    /**
     * Derivative in y axis direction.
     *
     * @param src {@code array} containing image data.
     * @param dest {@code array} in which calculations will be written.
     * @param width of the image.
     * @param height of the image.
     * @param pixelSize number of elements used per pixel.
     * @param column index of column being calculated.
     */
    private static void deriveY(float[] src, float[] dest, int width, int height, int pixelSize, int column) {
        int arrayWidth = pixelSize * width;
        int offset = arrayWidth + (column * pixelSize);

        for (int i = pixelSize, limit = height - 1; i < limit; i++) {
            for (int j = 0; j < pixelSize; j++) {
                dest[offset + j] = -src[offset + j - arrayWidth] + src[offset + j + arrayWidth];
            }

            offset += arrayWidth;
        }
    }

    /**
     * Derivative in x axis direction.
     *
     * @param src {@code array} containing image data.
     * @param dest {@code array} in which calculations will be written.
     * @param width of the image.
     * @param pixelSize number of elements used per pixel.
     * @param row index of row being calculated.
     */
    private static void deriveX(float[] src, float[] dest, int width, int pixelSize, int row) {
        int offset = pixelSize * width * row + pixelSize;

        for (int i = 1; i < width - 1; i++) {
            for (int j = 0; j < pixelSize; j++) {
                dest[offset] = -src[offset - pixelSize] + src[offset + pixelSize];
                offset++;
            }
        }
    }

    /**
     * Initializes the Queue with indexes.
     *
     * @param numberOfElement number of indexes to be added.
     * @param processors number of {@code killElement}s to add.
     * @param killElement element used to stop the thread.
     *
     * @return a new {@link BlockingQueue} containing indexes
     *         {@code 0} through {@code numberOfElements - 1}
     *         and {@code processors} elements used to stop
     *         the threads.
     */
    private static BlockingQueue<Integer> initializeQueue(int numberOfElement, int processors, int killElement) {
        BlockingQueue<Integer> queue = new ArrayBlockingQueue<>(numberOfElement + processors);

        for (int i = 0; i < numberOfElement; i++) {
            queue.offer(i);
        }

        for (int i = 0; i < processors; i++) {
            queue.offer(killElement);
        }

        return queue;
    }
}
