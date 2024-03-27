# SKA-low mini project: Gain Calibration


## Introduction

In the era of radio astronomy with initiatives like the Square Kilometer Array (SKA), the precision of instrument calibration is vital. This project focuses on implementing an algorithm for the retrieval of gain solutions for a single SKA-low station, which comprises 256
antennas. 

The repository contains a series of Python scripts that explore this calibration problem. The following section provides instructions on how to run the `main.py` script.


by **Max Talberg**

## Running the script locally

### Setting Up the Project

1. **Clone the repository:**
   - Clone the repository from GitLab:
     ```bash
     git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/A1_SKA_Assessment/mt942.git
     ```

2. **Set up the virtual environment:**

   - Create virtual environment:
     ```bash
     conda env create -f environment.yml
     ```
    - Activate virtual environment:
      ```bash
      conda activate ska-env
      ```
3. **Running the script:**

   - Run the main script:
     ```bash
     python src/main.py
     ```

### Notes

- Running the provided script will produce a sequence of plots:
    - The first two plots are part of the investigation into the most likely flash location analysis, addressing part (iii) of the problem.

    - The next four plots demonstrate the results of the NUTS for determining the lighthouse location based only on flash locations, corresponding to part (v). Results will be displayed in the terminal.

    - The next set of four plots illustrates the NUTS results using both flash locations and intensities, corresponding to part (vii). Results will be displayed in the terminal.

    - The final eight plots represent the appendix of the report. These plots are produced using ArviZ's and Corner's integrated plotting functions, they are accompanied by in-depth statistics in the terminal.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Use of generative tools

This project has utilised autogenerative tools in the development of documentation that is compatible with auto-documentation tools and the development of plotting functions.