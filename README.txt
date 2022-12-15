Brenden Collins // Nate Novak
CS7180 Advanced Perception
Fall 2022

An Implementation and Visualization of 'Attention Is All You Need'

OS: macOS Monterey 12.3.1
Hardware: Macbook Air M1 (2020)
Note: This code also runs on Nate's Ubuntu machine

Compile and run instructions:
No compilation needed

Python version: Python 3.9
From directory 'cs7180_finalproject':
  $ mkdir results
  $ python src/TrainTest.py 
      - calls training for the Transformer model 
      - unbatched, likely very slow
      - produces 'cs7180_finalproject/results/model.pth' when complete
  $ python src/Translate.py
      - loads model.pth and attempts to translate a sentence

P.S. I am including two .ipynb notebooks used to produce visualizations.
     If you wish to run these, they should be run one directory above cs7180_finalproject,
       so you should put the whole zipped submission directory inside another directory,
       and run the .ipynb files from that parent directory
       - This is because I left them out of the git repo, which is the main submission dir.
