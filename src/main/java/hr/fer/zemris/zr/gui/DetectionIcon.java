package hr.fer.zemris.zr.gui;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Line2D;
import java.net.URL;
import java.util.List;

/**
 * An {@link ImageIcon} used to display detected windows.
 *
 * @author Mateo Imbrišak
 */

public class DetectionIcon extends ImageIcon {

    private static final long serialVersionUID = 1359904102611456188L;

    /**
     * Default width of a detection window.
     */
    private static final int DEFAULT_WIDTH = 64;

    /**
     * Default height of a detection window.
     */
    private static final int DEFAULT_HEIGHT = 128;

    /**
     * Scaling factor used for scaled images.
     */
    private static final double DEFAULT_SCALE = 1.5;

    /**
     * Number of pixels used as a step of the window.
     */
    private static final int STEP_SIZE = 5;

    /**
     * Keeps image level and index pairs of detected images.
     */
    private final List<Integer> indexes;

    public DetectionIcon(List<Integer> indexes, String filename, String description) {
        super(filename, description);
        this.indexes = indexes;
    }

    public DetectionIcon(List<Integer> indexes, String filename) {
        super(filename);
        this.indexes = indexes;
    }

    public DetectionIcon(List<Integer> indexes, URL location, String description) {
        super(location, description);
        this.indexes = indexes;
    }

    public DetectionIcon(List<Integer> indexes, URL location) {
        super(location);
        this.indexes = indexes;
    }

    public DetectionIcon(List<Integer> indexes, Image image, String description) {
        super(image, description);
        this.indexes = indexes;
    }

    public DetectionIcon(List<Integer> indexes, Image image) {
        super(image);
        this.indexes = indexes;
    }

    public DetectionIcon(List<Integer> indexes, byte[] imageData, String description) {
        super(imageData, description);
        this.indexes = indexes;
    }

    public DetectionIcon(List<Integer> indexes, byte[] imageData) {
        super(imageData);
        this.indexes = indexes;
    }

    public DetectionIcon(List<Integer> indexes) {
        super();
        this.indexes = indexes;
    }

    @Override
    public synchronized void paintIcon(Component c, Graphics g, int x, int y) {
        super.paintIcon(c, g, x, y);

        final Graphics2D g2d = (Graphics2D) g;
        g2d.setColor(Color.GREEN);

        final int width = getIconWidth();
        final Line2D[] lines = new Line2D[4];

        for (int i = 0, size = indexes.size(); i < size; i += 2) {
            final double scale = Math.max(indexes.get(i) * DEFAULT_SCALE, 1);
            final int index = indexes.get(i + 1);

            final int windowRow = (width - DEFAULT_WIDTH + STEP_SIZE) / STEP_SIZE;
            final int rowOffset = index / windowRow * STEP_SIZE;
            final int columnOffset = index % windowRow * STEP_SIZE;

            lines[0] = new Line2D.Double(columnOffset * scale + x, rowOffset * scale + y,
                    columnOffset * scale + x, rowOffset * scale + DEFAULT_HEIGHT + y);
            lines[1] = new Line2D.Double(columnOffset * scale + x, rowOffset * scale + y,
                    columnOffset * scale + DEFAULT_WIDTH + x, rowOffset * scale + y);
            lines[2] = new Line2D.Double(columnOffset * scale + DEFAULT_WIDTH + x,
                    rowOffset * scale + DEFAULT_HEIGHT + y, columnOffset * scale + x,
                    rowOffset * scale + DEFAULT_HEIGHT + y);
            lines[3] = new Line2D.Double(columnOffset * scale + DEFAULT_WIDTH + x,
                    rowOffset * scale + DEFAULT_HEIGHT + y, columnOffset * scale + DEFAULT_WIDTH + x,
                    rowOffset * scale + y);

            for (Line2D line : lines) {
                g2d.draw(line);
            }
        }
    }
}
