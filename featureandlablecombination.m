clear all;
clc;
close all;

% Load labels and features
load('labelfc8.mat'); % Load labels
load('bestcottonclassificationnet45featuresfc8.mat'); % Load features

% Combine labels and features
features = bestcottonclassificationnet45featuresfc8;
labels = labelfc8;

% Save combined dataset to MAT file
save('bestCCnetfeatureswithlables45fc8.mat', 'features', 'labels');
