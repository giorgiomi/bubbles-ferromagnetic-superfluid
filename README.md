# Bubbles in a ferromagnetic superfluid
This is my Bachelor's thesis project in physics, investigating the characteristics of bubbles in a ferromagnetic superfluid, which are created through the process of false vacuum decay. The primary objective is to analyze the experimental data to identify and quantify the key properties of these bubbles. By examining the relationship between the experimental parameters and the observed bubble characteristics, this work aims to provide a deeper understanding of the underlying physical phenomena and contribute to the broader fields of ferromagnetism and superfluidity.

![poster](poster.png)

## Data analysis
Python code for the data analysis is in the folder `src/`. Make sure to have all the libraries listed in `requirements.txt` installed and then run
```
    python3 src/gather.py
```
where you may gather selected or all data. For some scripts one or the other is needed. Then run
```
    python3 src/<script_name>.py
```
The scripts do many different things and, at least for now, they are not properly commented, so handle them with care.

## Thesis
Thesis in LaTeX can be found in the folder `thesis/` or [here](https://github.com/giorgiomi/bubbles-ferromagnetic-superfluid/blob/master/thesis/main.pdf). Figures are not included in this repository.

## Presentation
Slides in LaTeX (beamer) can be found in the folder `presentation/` or [here](https://github.com/giorgiomi/bubbles-ferromagnetic-superfluid/blob/master/presentation/main.pdf). Figures are not included in this repository.