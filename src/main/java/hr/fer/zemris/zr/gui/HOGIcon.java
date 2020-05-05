package hr.fer.zemris.zr.gui;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.geom.Line2D;
import java.net.URL;

/**
 * A class used to draw HOG visualisation on an {@link ImageIcon}.
 *
 * @author Mateo Imbrišak
 */

public class HOGIcon extends ImageIcon {

    private static final long serialVersionUID = -3253212190599713896L;

    /**
     * Number of elements per row in {@link #featureVector}.
     */
    private static final int ROW_ELEMENTS = 16;

    /**
     * Number of elements per column in {@link #featureVector}.
     */
    private static final int COLUMN_ELEMENTS = 8;

    /**
     * Size of a block in pixels.
     */
    private static final int BLOCK_SIZE = 8;

    /**
     * Number of bins used in histograms.
     */
    private static final int BIN_SIZE = 9;

    /**
     * Initial angle used for rotation calculation.
     */
    private static final int INITIAL_ANGLE = 10;

    /**
     * Angle step used in rotation.
     */
    private static final int ANGLE_STEP = 20;

    /**
     * Keeps the feature vector drawn on the image.
     */
    private final double[] featureVector;

    public HOGIcon(double[] featureVector, String filename, String description) {
        super(filename, description);
        this.featureVector = featureVector;
    }

    public HOGIcon(double[] featureVector, String filename) {
        super(filename);
        this.featureVector = featureVector;
    }

    public HOGIcon(double[] featureVector, URL location, String description) {
        super(location, description);
        this.featureVector = featureVector;
    }

    public HOGIcon(double[] featureVector, URL location) {
        super(location);
        this.featureVector = featureVector;
    }

    public HOGIcon(double[] featureVector, Image image, String description) {
        super(image, description);
        this.featureVector = featureVector;
    }

    public HOGIcon(double[] featureVector, Image image) {
        super(image);
        this.featureVector = featureVector;
    }

    public HOGIcon(double[] featureVector, byte[] imageData, String description) {
        super(imageData, description);
        this.featureVector = featureVector;
    }

    public HOGIcon(double[] featureVector, byte[] imageData) {
        super(imageData);
        this.featureVector = featureVector;
    }

    public HOGIcon(double[] featureVector) {
        super();
        this.featureVector = featureVector;
    }

    @Override
    public synchronized void paintIcon(Component c, Graphics g, int x, int y) {
        super.paintIcon(c, g, x, y);

        final Graphics2D g2d = (Graphics2D) g;
        g2d.setColor(Color.RED);

        int index = 0;
        final int halfSize = BLOCK_SIZE / 2;

        for (int i = 0; i < ROW_ELEMENTS; i++) {
            for (int j = 0; j < COLUMN_ELEMENTS; j++) {
                double sum = 0;

                for (int k = 0; k < BIN_SIZE; k++) {
                    final double value = featureVector[index + k];
                    sum += value * value;
                }

                final double norm = Math.sqrt(sum);
                int currentAngle = INITIAL_ANGLE;

                for (int k = 0; k < BIN_SIZE; k++) {
                    final double halfLength = featureVector[index] / norm * halfSize;

                    final int xPosition = j * BLOCK_SIZE + halfSize + x;
                    final int yPosition = i * BLOCK_SIZE + halfSize + y;

                    final Line2D line = new Line2D.Double(xPosition - halfLength, yPosition,
                            xPosition + halfLength, yPosition);
                    AffineTransform at = AffineTransform.getRotateInstance(Math.toRadians(currentAngle),
                            xPosition, yPosition);
                    g2d.draw(at.createTransformedShape(line));
                    currentAngle += ANGLE_STEP;
                    index++;
                }
            }
        }
    }
}
