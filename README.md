# Not Hotdog powered by Akida One-Shot Learning

A classy example of using Akida's One-Shot Learning to determine if the object is a hotdog or not a hotdog.

This is an extremely loose example that takes only two images. One hotdog and one not hotdog. There is a training dataset out there specifically for the not hotdog app but I just wanted to experiment with a single image of a hotdog.


## Setting up the Akida development evironment

1. Go to `https://www.anaconda.com/download/` and download installer
2. Install anaconda `bash Anaconda-latest-Linux-x86_64.sh`
3. Create conda environment `conda create --name akida_env python=3.6`
4. Activate conda environement `conda activate akida_env`
5. Install python dependencies `pip install -r requirements.txt`

## Running and using the example

1. `python3 akida_not_hotdog.py`
2. Point webcam at object and press `space` to determine if it is a hotdog or not a hotdog
3. If you want to teach it more hotdogs or not hotdogs, point webcam at something and press `y` if its a hotdog or `n` if its not hotdog

## As seen in the hilarious TV Show `Silicon Valley`

The concept is taken from the TV show

[![Not Hotdog](http://img.youtube.com/vi/pqTntG1RXSY/0.jpg)](https://youtu.be/pqTntG1RXSY "Not Hotdog")

### Read More

Read all the documentation at https://doc.brainchipinc.com