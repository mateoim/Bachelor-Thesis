package hr.fer.zemris.zr.gui;

import hr.fer.zemris.zr.HOGCalculator;
import hr.fer.zemris.zr.data.HOGImage;
import hr.fer.zemris.zr.util.Algorithms;
import hr.fer.zemris.zr.util.ThreadsJobs;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.FileVisitor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

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
     * Keeps the {@link SVM} model to be trained and later used to detect pedestrians.
     */
    private final SVM model = SVM.create();

    /**
     * Used to display {@link #loadedImage}.
     */
    private final JLabel imageLabel = new JLabel("", SwingConstants.CENTER);

    /**
     * A {@link JCheckBox} used to control vector calculation parallelization.
     */
    private final JCheckBox parallelCheckbox = new JCheckBox("Parallel vector calculation", true);

    /**
     * A {@link JCheckBox} used to control histogram calculation and normalization parallelization.
     */
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

        final JButton train = new JButton("Train model");
        buttonPanel.add(train);

        final JButton window = new JButton("Sliding window");
        buttonPanel.add(window);
        window.setEnabled(false);

        train.addActionListener(e -> {
            train.setEnabled(false);
            trainModel();
            window.setEnabled(true);
        });

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

        final JMenu file = new JMenu("File");
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
        final float[] featureVector = image.calculateFeatureVector();

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

        float[][][] featureVectors = new float[pyramid.size()][][];

        int index = 0;

        for (HOGImage hogImage : pyramid) {
            featureVectors[index++] = hogImage.slideWindow(parallelCheckbox.isSelected(),
                    parallelChildrenCheckbox.isSelected());
        }

        List<Integer> list = new ArrayList<>();
        int outerCounter = 0;
        int counter;

        for (float[][] image : featureVectors) {
            counter = 0;
            for (float[] vector : image) {
                Mat toPredict = new Mat(1, vector.length, CvType.CV_32FC1);
                toPredict.put(0, 0, vector);

                if (model.predict(toPredict) == 1 && outerCounter == 0) {
                    list.add(outerCounter);
                    list.add(counter);
                }
                counter++;
            }
            outerCounter++;
        }

        long endTime = System.nanoTime();

        System.out.println((endTime - startTime) / 1_000_000_000d);

        imageLabel.setIcon(new DetectionIcon(list, loadedImage));
    }

    /**
     * Used to initialize {@link #model} training.
     */
    private void trainModel() {
        final JFileChooser jfc = new JFileChooser();
        jfc.setDialogTitle("Select folder with training examples");
        jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

        if (jfc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }

        final Path src = jfc.getSelectedFile().toPath();

        final Path positive = src.resolve("pos");
        final Path negative = src.resolve("neg");

        if (!Files.exists(positive) || !Files.exists(negative)) {
            JOptionPane.showOptionDialog(
                    this,
                    "Selected directory must contain \"pos\" and \"neg\" directories.",
                    "Error",
                    JOptionPane.DEFAULT_OPTION,
                    JOptionPane.ERROR_MESSAGE,
                    null, null, null);
            return;
        }

        final float[][] positiveVectors = calculateSubset(positive);

        if (positiveVectors == null) return;

        final float[][] negativeVectors = calculateSubset(negative);

        if (negativeVectors == null) return;

        trainSVM(positiveVectors, negativeVectors);
    }

    /**
     * Used internally to train {@link #model}.
     *
     * @param positiveVectors vectors calculated from positive examples.
     * @param negativeVectors vectors calculated from negative examples.
     */
    private void trainSVM(float[][] positiveVectors, float[][] negativeVectors) {
        final int totalSize = positiveVectors.length + negativeVectors.length;
        final int[] labels = new int[totalSize];
        final float[] values = new float[totalSize * positiveVectors[0].length];

        int index = 0;
        int labelIndex = 0;

        for (float[] vector : positiveVectors) {
            labels[labelIndex++] = 1;

            for (float value : vector) {
                values[index++] = value;
            }
        }

        for (float[] vector : negativeVectors) {
            labels[labelIndex++] = -1;

            for (float value : vector) {
                values[index++] = value;
            }
        }

        Mat trainingDataMat = new Mat(totalSize, positiveVectors[0].length, CvType.CV_32FC1);
        trainingDataMat.put(0, 0, values);
        Mat labelsMat = new Mat(totalSize, 1, CvType.CV_32SC1);
        labelsMat.put(0, 0, labels);

        model.setType(SVM.C_SVC);
        model.setKernel(SVM.LINEAR);
        model.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 10_000, 1e-6));
        model.train(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);
    }

    /**
     * Used to calculate feature vectors for a subset contained in the given directory.
     *
     * @param path to the directory containing the subset.
     *
     * @return an {@code array} containing calculated feature vectors.
     */
    private float[][] calculateSubset(Path path) {
        final ImageVisitor visitor = new ImageVisitor();

        try {
            Files.walkFileTree(path, visitor);
        } catch (IOException exc) {
            JOptionPane.showOptionDialog(
                    this,
                    "Error opening a file",
                    "Error",
                    JOptionPane.DEFAULT_OPTION,
                    JOptionPane.ERROR_MESSAGE,
                    null, null, null);
            return null;
        }

        final BlockingQueue<BufferedImage> queue = visitor.createQueue();
        final int processors = Runtime.getRuntime().availableProcessors();
        final int size = queue.size() - processors;

        final float[][] vectors = new float[size][];

        final BlockingQueue<float[]> calculatedVectors = new ArrayBlockingQueue<>(size);

        final Thread[] threads = new Thread[processors];

        for (int i = 0; i < processors; i++) {
            Thread thread = new Thread(() -> ThreadsJobs.positiveCalculationThread(queue, calculatedVectors,
                    visitor.killElement));
            threads[i] = thread;
            thread.start();
        }

        Algorithms.joinThreads(threads);

        int index = 0;

        for (float[] vector : calculatedVectors) {
            vectors[index++] = vector;
        }

        return vectors;
    }

    /**
     * A {@link FileVisitor} used to read images.
     */
    private static class ImageVisitor implements FileVisitor<Path> {

        /**
         * Keeps read images.
         */
        private final List<BufferedImage> images = new LinkedList<>();

        /**
         * Element used to stop threads.
         */
        private final BufferedImage killElement = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);

        /**
         * Creates a {@link BlockingQueue} containing read images and kill elements corresponding to
         * the number of available processors.
         *
         * @return created {@link BlockingQueue}.
         */
        public BlockingQueue<BufferedImage> createQueue() {
            final int processors = Runtime.getRuntime().availableProcessors();
            BlockingQueue<BufferedImage> queue = new ArrayBlockingQueue<>(images.size() + processors);

            for (BufferedImage image : images) {
                queue.offer(image);
            }

            for (int i = 0; i < processors; i++) {
                queue.offer(killElement);
            }

            return queue;
        }

        @Override
        public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
            return FileVisitResult.CONTINUE;
        }

        @Override
        public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
            BufferedImage image = ImageIO.read(file.toFile());

            if (image != null) {
                images.add(image);
            }

            return FileVisitResult.CONTINUE;
        }

        @Override
        public FileVisitResult visitFileFailed(Path file, IOException exc) throws IOException {
            return FileVisitResult.CONTINUE;
        }

        @Override
        public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
            return FileVisitResult.CONTINUE;
        }
    }
}
