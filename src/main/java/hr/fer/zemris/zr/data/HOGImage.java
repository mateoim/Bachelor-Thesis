package hr.fer.zemris.zr.data;

import hr.fer.zemris.zr.util.Algorithms;

import java.awt.image.BufferedImage;

/**
 * A class that keeps data about histogram of gradients of the given image.
 *
 * @author Mateo Imbrišak
 */

public class HOGImage {

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
}
