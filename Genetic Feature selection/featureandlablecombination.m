clear all;
clc;
close all;

% Load labels and features
load('label.mat'); % Load labels
load('bestcottonclassificationnetfeatures.mat'); % Load features

% Combine labels and features
features = bestcottonclassificationnetfeatures;
labels = YTrain;

% Save combined dataset to MAT file
save('bestCCnetfeatureswithlables.mat', 'features', 'labels');
