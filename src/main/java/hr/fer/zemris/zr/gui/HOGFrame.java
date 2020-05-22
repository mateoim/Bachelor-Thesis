package hr.fer.zemris.zr.gui;

import hr.fer.zemris.zr.HOGCalculator;
import hr.fer.zemris.zr.data.HOGImage;
import hr.fer.zemris.zr.util.Algorithms;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Class that represents GUI of {@link HOGCalculator}.
 *
 * @author Mateo Imbrišak
 */

public class HOGFrame extends JFrame {

    /**
     * Default scaling factor used to create a pyramid.
     */
    private static final double SCALING_FACTOR = 1.5;

    /**
     * Required width of image to be used with the algorithm.
     */
    private static final int DEFAULT_WIDTH = 64;

    /**
     * Required height of image to be used with the algorithm.
     */
    private static final int DEFAULT_HEIGHT = 128;

    /**
     * Keeps currently loaded image.
     */
    private BufferedImage loadedImage;

    /**
     * Keeps loaded image as an OpenCV {@link Mat}.
     */
    private Mat image;

    /**
     * Used to display {@link #loadedImage}.
     */
    private final JLabel imageLabel = new JLabel("", SwingConstants.CENTER);

    private final JCheckBox parallelCheckbox = new JCheckBox("Parallel vector calculation", true);

    private final JCheckBox parallelChildrenCheckbox =
            new JCheckBox("Parallel histogram calculation and normalization");

    /**
     * Action used to open an image.
     */
    private final Action openAction = new AbstractAction() {
        private static final long serialVersionUID = 4725649992068113114L;

        @Override
        public void actionPerformed(ActionEvent e) {
            open();
        }
    };

    /**
     * Default constructor that initializes the GUI and displays the window.
     */
    public HOGFrame() {
        setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);

        initGUI();
        setVisible(true);
    }

    /**
     * Used internally to initialize components.
     */
    private void initGUI() {
        add(imageLabel, BorderLayout.CENTER);
        JPanel controlsPanel = new JPanel(new BorderLayout());
        add(controlsPanel, BorderLayout.SOUTH);

        final JPanel buttonPanel = new JPanel();
        controlsPanel.add(buttonPanel, BorderLayout.NORTH);

        final JButton dx = new JButton("dx");
        dx.addActionListener((l) -> deriveImage('x'));

        final JButton dy = new JButton("dy");
        dy.addActionListener((l) -> deriveImage('y'));

        buttonPanel.add(dx);
        buttonPanel.add(dy);

        final JButton magnitude = new JButton("magnitude");
        magnitude.addActionListener((l) -> calculateMagnitude());

        final JButton angle = new JButton("angle");
        angle.addActionListener((l) -> calculateAngle());

        buttonPanel.add(magnitude);
        buttonPanel.add(angle);

        final JButton hog = new JButton("HOG visualization");
        buttonPanel.add(hog);

        hog.addActionListener(e -> calculateHOG());

        final JButton window = new JButton("Sliding window");
        buttonPanel.add(window);

        window.addActionListener(e -> slidingWindow());

        JPanel checkboxPanel = new JPanel();
        controlsPanel.add(checkboxPanel, BorderLayout.SOUTH);
        checkboxPanel.setBorder(new TitledBorder("Parallelization settings"));

        checkboxPanel.add(parallelCheckbox);
        checkboxPanel.add(parallelChildrenCheckbox);

        initMenuBar();

        pack();
    }

    /**
     * Used internally to initialize the menu bar.
     */
    private void initMenuBar() {
        final JMenuBar menu = new JMenuBar();

        final JMenu file = new JMenu("file");
        file.add(openAction);

        openAction.putValue(Action.NAME, "Open");
        openAction.putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("control O"));
        openAction.putValue(Action.MNEMONIC_KEY, KeyEvent.VK_O);

        menu.add(file);
        setJMenuBar(menu);
    }

    /**
     * Implementation of {@link #openAction}. Used to choose
     * file to open and show error dialog if the file cannot be opened.
     */
    private void open() {
        final JFileChooser jfc = new JFileChooser();
        jfc.setDialogTitle("Open file");

        if (jfc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }

        final Path src = jfc.getSelectedFile().toPath();

        try {
            loadedImage = ImageIO.read(src.toFile());
            imageLabel.setIcon(new ImageIcon(loadedImage));
            pack();
            image = Imgcodecs.imread(src.toString());
        } catch (IOException | NullPointerException exc) {
            JOptionPane.showOptionDialog(
                    this,
                    "Error while opening " + src.toString(),
                    "Error",
                    JOptionPane.DEFAULT_OPTION,
                    JOptionPane.ERROR_MESSAGE,
                    null, null, null);
        }
    }

    /**
     * Calculates derivative of {@link #loadedImage} and displays it on {@link #imageLabel}.
     *
     * @param direction of derivative.
     */
    private void deriveImage(char direction) {
        if (loadedImage == null) {
            return;
        }

        final int width = loadedImage.getWidth();
        final int height = loadedImage.getHeight();

        final int pixelCount = loadedImage.getAlphaRaster() == null ? 3 : 4;

        final float[] data = Algorithms.getDataFloat(loadedImage);
        final float[] derived = Algorithms.derive(data, direction, width, height, pixelCount, true);

        final BufferedImage result = Algorithms.convertToImage(derived, width, height, loadedImage.getType());

        imageLabel.setIcon(new ImageIcon(result));
    }

    /**
     * Calculates image magnitude and displays results on {@link #imageLabel}.
     */
    private void calculateMagnitude() {
        if (loadedImage == null) {
            return;
        }

        final int width = loadedImage.getWidth();
        final int height = loadedImage.getHeight();

        final int pixelCount = loadedImage.getAlphaRaster() == null ? 3 : 4;

        final float[] data = Algorithms.getDataFloat(loadedImage);

        final float[] magnitude = Algorithms.calculateMagnitude(
                Algorithms.derive(data, 'x', width, height, pixelCount, true),
                Algorithms.derive(data, 'y', width, height, pixelCount, true));

        final BufferedImage img = Algorithms.convertToImage(magnitude, width, height, loadedImage.getType());

        imageLabel.setIcon(new ImageIcon(img));
    }

    /**
     * Calculates image angle and displays results on {@link #imageLabel}.
     */
    private void calculateAngle() {
        if (loadedImage == null) {
            return;
        }

        final int width = loadedImage.getWidth();
        final int height = loadedImage.getHeight();

        final int pixelCount = loadedImage.getAlphaRaster() == null ? 3 : 4;

        final float[] data = Algorithms.getDataFloat(loadedImage);

        final float[] angle = Algorithms.calculateAngle(
                Algorithms.derive(data, 'x', width, height, pixelCount, true),
                Algorithms.derive(data, 'y', width, height, pixelCount, true));

        final BufferedImage img = Algorithms.convertToImage(angle, width, height, loadedImage.getType());

        imageLabel.setIcon(new ImageIcon(img));
    }

    /**
     * Calculates HOG and displays visualisation on {@link #imageLabel}.
     */
    private void calculateHOG() {
        if (loadedImage == null) {
            return;
        }

        final HOGImage image = new HOGImage(loadedImage);
        final double[] featureVector = image.calculateFeatureVector();

        imageLabel.setIcon(new HOGIcon(featureVector, loadedImage));
    }

    /**
     * Creates an image pyramid and runs the sliding window algorithm.
     */
    private void slidingWindow() {
        if (loadedImage == null) {
            return;
        }

        long startTime = System.nanoTime();

        List<HOGImage> pyramid = Algorithms.createPyramid(image, SCALING_FACTOR, DEFAULT_HEIGHT, DEFAULT_WIDTH);

        double[][][] featureVectors = new double[pyramid.size()][][];

        int index = 0;

        for (HOGImage hogImage : pyramid) {
            featureVectors[index++] = hogImage.slideWindow(parallelCheckbox.isSelected(),
                    parallelChildrenCheckbox.isSelected());
        }

        long endTime = System.nanoTime();

        System.out.println((endTime - startTime) / 1_000_000_000d);
    }
}
