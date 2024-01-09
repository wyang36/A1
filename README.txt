To run the code and reproduce the results for Assignment 1, do the following:
1. Download and unzip the files from: https://drive.google.com/file/d/1kRCITCrIVeIBXEOgEQ4N0cdCqcQuTS1W/view?usp=sharing
2. If not done so already, install miniconda (https://docs.conda.io/en/latest/miniconda.html) on your machine
3. cd into the directory
4. Create and activate the conda environment using:
	conda env create -f environment.yml
	conda activate ml
5. The python files are named as algorithm_dataset.py, example: dt_heart.py, dt_mbti.py. Each file is individually runnable. It will print the GridSearchCV results and final test results and export any plots to ./images folder. The plots used in the analysis are already present but running the code files will overwrite them. To run a code file:
	python dt_heart.py