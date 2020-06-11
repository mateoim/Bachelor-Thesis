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
     * Positive train example directory name.
     */
    private static final String TRAIN_POSITIVE = "pos";

    /**
     * Negative train example directory name.
     */
    private static final String TRAIN_NEGATIVE = "neg";

    /**
     * Positive test example directory name.
     */
    private static final String TEST_POSITIVE = "posTest";

    /**
     * Negative test example directory name.
     */
    private static final String TEST_NEGATIVE = "negTest";

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
     * Keeps {@link DetectionIcon}s for all levels.
     */
    private final List<DetectionIcon> detectionList = new ArrayList<>();

    /**
     * Keeps track of the current index in {@link #detectionList}.
     */
    private int currentIndex = 0;

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

        final JButton magnitude = new JButton("Magnitude");
        magnitude.addActionListener((l) -> calculateMagnitude());

        final JButton angle = new JButton("Angle");
        angle.addActionListener((l) -> calculateAngle());

        buttonPanel.add(magnitude);
        buttonPanel.add(angle);

        final JButton hog = new JButton("HOG visualization");
        buttonPanel.add(hog);

        hog.addActionListener(e -> calculateHOG());

        final JButton train = new JButton("Train model");
        buttonPanel.add(train);

        final JButton test = new JButton("Test model");
        buttonPanel.add(test);
        test.setEnabled(false);

        final JButton window = new JButton("Run detection");
        buttonPanel.add(window);
        window.setEnabled(false);

        train.addActionListener(e -> {
            train.setEnabled(false);
            if (!trainModel()) {
                train.setEnabled(true);
            } else {
                window.setEnabled(true);
                test.setEnabled(true);
            }
        });

        window.addActionListener(e -> slidingWindow());
        test.addActionListener(e -> testModel());

        JPanel checkboxPanel = new JPanel();
        controlsPanel.add(checkboxPanel, BorderLayout.SOUTH);
        checkboxPanel.setBorder(new TitledBorder("Parallelization settings"));

        checkboxPanel.add(parallelCheckbox);
        checkboxPanel.add(parallelChildrenCheckbox);

        final JButton previous = new JButton("Previous");
        buttonPanel.add(previous);

        previous.addActionListener(l -> {
            if (detectionList.isEmpty()) return;
            currentIndex = Math.max(0, currentIndex - 1);
            imageLabel.setIcon(detectionList.get(currentIndex));
        });

        final JButton next = new JButton("Next");
        buttonPanel.add(next);

        next.addActionListener(l -> {
            if (detectionList.isEmpty()) return;
            currentIndex = Math.min(detectionList.size() - 1, currentIndex + 1);
            imageLabel.setIcon(detectionList.get(currentIndex));
        });

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
            currentIndex = 0;
            detectionList.clear();
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
        final float[] featureVector;

        try {
            final float[][] histograms = image.calculateHistogram();
            featureVector = new float[histograms.length * histograms[0].length];
            int index = 0;

            for (float[] histogram : histograms) {
                for (float value : histogram) {
                    featureVector[index++] = value;
                }
            }
        } catch (UnsupportedOperationException exc) {
            JOptionPane.showOptionDialog(
                    this,
                    "Only available for images with width 64 and height 128.",
                    "Error",
                    JOptionPane.DEFAULT_OPTION,
                    JOptionPane.ERROR_MESSAGE,
                    null, null, null);
            return;
        }

        imageLabel.setIcon(new HOGIcon(featureVector, loadedImage));
    }

    /**
     * Creates an image pyramid and runs the sliding window algorithm.
     */
    private void slidingWindow() {
        if (loadedImage == null) {
            return;
        }

        detectionList.clear();
        currentIndex = 0;

        final float[][][] featureVectors = pyramidDetection(image);

        List<Integer> list;
        BufferedImage currentImage = loadedImage;
        Mat currentMat = image;
        int counter;
        final int colorType = loadedImage.getType();

        for (float[][] image : featureVectors) {
            counter = 0;
            list = new LinkedList<>();
            for (float[] vector : image) {
                Mat toPredict = new Mat(1, vector.length, CvType.CV_32FC1);
                toPredict.put(0, 0, vector);

                if (model.predict(toPredict) == 1) {
                    list.add(counter);
                }
                counter++;
            }
            detectionList.add(new DetectionIcon(list, currentImage));
            currentMat = Algorithms.scaleMat(currentMat, SCALING_FACTOR);
            currentImage = Algorithms.toBufferedImage(currentMat, colorType);
        }

        imageLabel.setIcon(detectionList.get(currentIndex));
    }

    /**
     * Calculates feature vectors for all images in an image pyramid for the given image.
     *
     * @param image for whose pyramid feature vectors will be calculated.
     *
     * @return feature vectors for all images in the pyramid.
     */
    private float[][][] pyramidDetection(Mat image) {
        final List<HOGImage> pyramid = Algorithms.createPyramid(image, SCALING_FACTOR, DEFAULT_HEIGHT, DEFAULT_WIDTH);

        final float[][][] featureVectors = new float[pyramid.size()][][];

        int index = 0;

        for (HOGImage hogImage : pyramid) {
            featureVectors[index++] = hogImage.slideWindow(parallelCheckbox.isSelected(),
                    parallelChildrenCheckbox.isSelected());
        }

        return featureVectors;
    }

    /**
     * Used to initialize {@link #model} training.
     *
     * @return {@code true} if the model was trained successfully,
     *         otherwise {@code false}.
     */
    private boolean trainModel() {
        final JFileChooser jfc = new JFileChooser();
        jfc.setDialogTitle("Select folder with training examples");
        jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

        if (jfc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return false;
        }

        final Path src = jfc.getSelectedFile().toPath();

        final Path positive = src.resolve(TRAIN_POSITIVE);
        final Path negative = src.resolve(TRAIN_NEGATIVE);

        if (!Files.exists(positive) || !Files.exists(negative)) {
            JOptionPane.showOptionDialog(
                    this,
                    "Selected directory must contain \"" + TRAIN_POSITIVE + "\" and \""
                            + TRAIN_NEGATIVE+ "\" directories.",
                    "Error",
                    JOptionPane.DEFAULT_OPTION,
                    JOptionPane.ERROR_MESSAGE,
                    null, null, null);
            return false;
        }

        final float[][] positiveVectors = calculateSubset(positive, true);

        if (positiveVectors == null) return false;

        final float[][] negativeVectors = calculateSubset(negative, false);

        if (negativeVectors == null) return false;

        trainSVM(positiveVectors, negativeVectors);
        return true;
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
     * Used to test the trained model on a test dataset and evaluate accuracy.
     */
    private void testModel() {
        final JFileChooser jfc = new JFileChooser();
        jfc.setDialogTitle("Select folder with test examples");
        jfc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

        if (jfc.showOpenDialog(this) != JFileChooser.APPROVE_OPTION) {
            return;
        }

        final Path src = jfc.getSelectedFile().toPath();

        final Path positivePath = src.resolve(TEST_POSITIVE);
        final Path negativePath = src.resolve(TEST_NEGATIVE);

        if (!Files.exists(positivePath) || !Files.exists(negativePath)) {
            JOptionPane.showOptionDialog(
                    this,
                    "Selected directory must contain \"" + TEST_POSITIVE + "\" and \"" +
                            TEST_NEGATIVE + "\" directories.",
                    "Error",
                    JOptionPane.DEFAULT_OPTION,
                    JOptionPane.ERROR_MESSAGE,
                    null, null, null);
            return;
        }

        final List<Mat> positiveImages = loadImages(positivePath);

        if (positiveImages == null) return;

        final List<Mat> negativeImages = loadImages(negativePath);

        if (negativeImages == null) return;

        final int[] positive = detectObjects(positiveImages);
        final int[] negative = detectObjects(negativeImages);

        JOptionPane.showOptionDialog(
                this,
                "True positive: " + positive[0] + " False negative: " + positive[1]
                + "\nFalse positive: " + negative[0] + " True negative: " + negative[1],
                "Results",
                JOptionPane.DEFAULT_OPTION,
                JOptionPane.INFORMATION_MESSAGE,
                null, null, null);
    }

    /**
     * Loads images as {@link Mat} objects from the given path.
     *
     * @param path from which the images should be loaded.
     *
     * @return a {@link List} of all found images.
     */
    private List<Mat> loadImages(Path path) {
        final MatVisitor visitor = new MatVisitor();

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

        return visitor.images;
    }

    /**
     * Runs object detection on the given images.
     *
     * @param images {@link List} of images being analyzed.
     *
     * @return an {@code array} containing number of positive classifications
     *         at index {@code 0} and negative classifications at index {@code 1}.
     */
    private int[] detectObjects(List<Mat> images) {
        int positive = 0;
        int negative = 0;

        for (Mat image : images) {
            final float[][][] featureVectors = pyramidDetection(image);
            boolean found = false;

            for (float[][] outer : featureVectors) {
                for (float[] vector : outer) {
                    Mat toPredict = new Mat(1, vector.length, CvType.CV_32FC1);
                    toPredict.put(0, 0, vector);

                    if (model.predict(toPredict) == 1) {
                        positive++;
                        found = true;
                        break;
                    }
                }

                if (found) break;
            }

            if (!found) {
                negative++;
            }
        }

        return new int[] {positive, negative};
    }

    /**
     * Used to calculate feature vectors for a subset contained in the given directory.
     *
     * @param path to the directory containing the subset.
     * @param positive used to indicate whether positive or negative examples are being calculated.
     *
     * @return an {@code array} containing calculated feature vectors.
     */
    private float[][] calculateSubset(Path path, boolean positive) {
        final ImageVisitor visitor = new ImageVisitor(positive);

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

        final BlockingQueue<float[]> calculatedVectors = new ArrayBlockingQueue<>(size *
                (positive ? 1 : ThreadsJobs.NUMBER_OF_SAMPLES));

        final Thread[] threads = new Thread[processors];

        for (int i = 0; i < processors; i++) {
            Thread thread;
            if (positive) {
                thread = new Thread(() -> ThreadsJobs.positiveCalculationThread(queue, calculatedVectors,
                        visitor.killElement));
            } else {
                thread = new Thread(() -> ThreadsJobs.negativeCalculationThread(queue, calculatedVectors,
                        visitor.killElement));
            }

            threads[i] = thread;
            thread.start();
        }

        Algorithms.joinThreads(threads);

        final float[][] vectors = new float[calculatedVectors.size()][];

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
         * Keeps track of whether positive examples are being read or not.
         */
        private final boolean positive;

        /**
         * Keeps read images.
         */
        private final List<BufferedImage> images = new LinkedList<>();

        /**
         * Element used to stop threads.
         */
        private final BufferedImage killElement = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);

        /**
         * Default constructor.
         *
         * @param positive used to define whether examples should be scaled to {@code 64 x 128}.
         */
        public ImageVisitor(boolean positive) {
            this.positive = positive;
        }

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
                final int width = image.getWidth();
                if (positive && width != DEFAULT_WIDTH) {
                    final Mat mat = Imgcodecs.imread(file.toString());
                    image = Algorithms.scaleImage(mat, width / (double) DEFAULT_WIDTH, image.getType());
                }

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

    /**
     * A {@link FileVisitor} used to read images as {@link Mat} objects.
     */
    private static class MatVisitor implements FileVisitor<Path> {

        /**
         * Keeps read images.
         */
        private final List<Mat> images = new LinkedList<>();

        @Override
        public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
            return FileVisitResult.CONTINUE;
        }

        @Override
        public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
            images.add(Imgcodecs.imread(file.toString()));
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
