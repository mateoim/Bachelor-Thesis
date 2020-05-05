package hr.fer.zemris.zr;

import hr.fer.zemris.zr.gui.HOGFrame;

import javax.swing.SwingUtilities;

/**
 * Class used to start the program.
 *
 * @author Mateo Imbrišak
 */

public class HOGCalculator {

    /**
     * Don't let anyone instantiate this class.
     */
    private HOGCalculator() {}

    /**
     * Used to start the program.
     *
     * @param args nothing.
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(HOGFrame::new);
    }
}
