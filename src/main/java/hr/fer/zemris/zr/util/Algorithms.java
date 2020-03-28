package hr.fer.zemris.zr.util;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

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
}
