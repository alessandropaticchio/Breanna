# Breanna

The goal of our project is to build a predictive tool to predict the effectiveness of banner design. One thing we noticed from our literature review is that, to learn a model of how the visual aspect influence the performance of a web banner, we typically need a large number of banner designs. 

The banner design should be diverse enough so that patterns will emerge. However, if we look at the banners used for a single campaign, we can see that usually they are visually very similar to each other. So to learn meaningful pattern of how the visual feature influence the banner's effectiveness, we need to collect data of various campaigns. Another thing we notice during our project is that the data for a single campaign is scattered in many place: the design of banners are kept as images and JavaScript, the user interaction data is kept as several different kinds of log files. 

Actually, in this project, we spent most of our time cleaning the data and just holding everything together rather than doing modeling. Therefore, we think that being able to organize data in a way that is easily accessable for analysis is crutial to make the analysis of banner designs possible. Ideally, we want this to be done automatically. This is the motivation for Breanna.

Installing Breanna

1. clone this repository by `git clone git@github.com:Harvard-IACS/2019-AC297rs-MSL.git`
2. Breanna lives in the `deliverable` folder. go to that folder by `cd 2019-AC297rs-MSL/deliverable`
3. (optional) install virtualenv by `pip install virtualenv` and create a virtual environment by `virtualenv msl_env`. Activate this environment by `source msl_env/bin/activate`. (use `deactivate` to deactivate the this environment later.)
4. install the dependensies by `pip install -r requirements.txt`
5. install `breanna` by `pip install -e .`

See `Documentation for Breanna` for how to use Breanna.

A more extensive description of the project can be found here: https://alefac912.github.io/MSL-datashack2019/
