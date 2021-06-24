# Real-time_Traffic_Detection_from_Tweets
BTech CSE Final Year Project 2020-21


A deep learning based real-time tarffic prediction model that is used to detect the road traffic using twitter data.
 
## Modules used

### For CNN model creation

* Google Colab - (Python3, gensim, Scikit, Pandas, tensorflow, Numpy, matplotlib)

### For creating website

* django
* pandas
* datetime
* time
* tensorflow
* requests
* json
* tweepy
* numpy
* pytz
* csv
* HTML5-CSS

## Steps to run .ipynb files

* Use [Google Colab](https://colab.research.google.com/notebooks/) to run .ipynb files in the cloud.

## Steps to deploy model in local system

* Open cmd
* Create virtual environment
* Install dependencies mentioned above for django
* Change directory to the place where the code is saved(directory -> buttonpython)
* run "python3 manage.py runserver 127.0.0.1:5000". 
* Copy the local link.
* Paste it on browser to deploy locally

### Step 1 — Install virtualenv

To install virtualenv, we will use the pip3 command, as shown below:

    pip3 install virtualenv

 

Once it is installed, run a version check to verify that the installation has completed successfully:

    virtualenv --version

### Step 2 — Install Django

Install Django within a virtualenv.

This is ideal for when you need your version of Django to be isolated from the global environment of your server.

While in the server’s home directory, we have to create the directory that will contain our Django application. Run the following command to create a directory called django-apps, or another name of your choice. Then navigate to the directory.

    mkdir buttonpython
    cd buttonpython

While inside the django-apps directory, create your virtual environment. Let’s call it cnn_env.

    virtualenv cnn_env

 Now, activate the virtual environment with the following command:

    . cnn_env/bin/activate

Within the environment, install the Django package using pip. Installing Django allows us to create and run Django applications. To learn more about Django, read our tutorial series on Django Development.

    pip install django

 Once installed, verify your Django installation by running a version check:

    django-admin --version
    
### Step 3 : Install the following.

* nltk
* pandas
* geopy
* tweepy
   

## Note:-

* If changes in dataset is made, model might give out undesirable outputs.
* Install all python modules mentioned above before running the project.
* tweet_final.csv gives output after prediction based on real-time  data in csv format.

