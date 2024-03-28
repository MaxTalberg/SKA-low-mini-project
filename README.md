# SKA-low mini project: Gain Calibration


## Introduction

In the era of radio astronomy with initiatives like the Square Kilometer Array (SKA), the precision of instrument calibration is vital. This project focuses on implementing an algorithm for the retrieval of gain solutions for a single SKA-low station, which comprises 256 antennas. 

The repository contains a series of Python scripts that explore this calibration problem. The following section provides instructions on how to run the `main.py` script.


by **Max Talberg**

## Running the script on Docker

### Setting Up the Project

1. **Clone the repository:**
   - Clone the repository from GitLab:
     ```bash
     git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/A1_SKA_Assessment/mt942.git
     ```

2. **Build the Docker image:**
    - Navigate to the project directory:
      ```bash
      cd mt942
      ```
   - Build image:
     ```bash
     docker build -t ska-project .
     ```

3. **Running the script:**

   - Run the main script:
     ```bash
     docker run -v host_directory:/app/plots ska-project
     ```
        - Replace `host_directory` with the path to the directory where you want to save the plots, for example: `/path/to/plots` and all the images will be saved into a folder named `plots`, information acompanying will be in the terminal output.


## Running the script locally

### Setting Up the Project

1. **Clone the repository:**
   - Clone the repository from GitLab:
     ```bash
     git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/A1_SKA_Assessment/mt942.git
     ```

2. **Set up the virtual environment:**
   - Navigate to the project directory:
     ```bash
       cd mt942
       ```
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
    - The first two depict the variability in the EEPs for each individual antenna element and the smoothed average response (2).

    - The next four plots illsutrate the absolute error in gain, amplitdue and phase for two different model matricies $M_{EEPs}$ and $M_{AEP}$. These results are plotted for two versions of the StEFCal algorithm (3/4).

    - The next two plots illustrate the beamformed power patterns using the different gain solutions from the StEFCal algorithm, further information about the algorithms will be printed to the terminal (5).

    - The final two plots depcit a 3D plot in sine-cosine coordiantes of the beamformed power patterns using the most accurate calibrated gain solution (6).

## Testing

### Running Unit Tests

1. **Navigate to the project directory `mt942 and run Unit Tests:**

      ```bash
       pytest
     ```
    
## Documentation

### Generating Documentation
1. **Navigate to the `docs` directory:**

      ```bash
       cd docs
     ```
2. **Generate the documentation:**

      ```bash
       make html
     ```
3. **Open the documentation:**

      ```bash
       open build/html/index.html

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Use of generative tools

This project has utilised auto-generative tools in the development of documentation that is compatible with auto-documentation tools, latex formatting and the development of plotting functions. 

Example prompts used for this project:
- Generate doc-strings in NumPy format for this function.
- Generate Latex code for a subplot.
- Generate Latex code for a 3 by 3 matrix.
- Generate Python code for a 2 by 1 subplot.
