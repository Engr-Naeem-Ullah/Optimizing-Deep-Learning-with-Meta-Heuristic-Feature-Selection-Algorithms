clear all;
clc;
close all;

%uploading testing dataset for feature extraction purpose
imds = imageDatastore('G:\upcoming research\Cotton leaves Dataset for diseases classification using EfficienNet Model\final results\Testing', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

numTrainImages = numel(imds.Labels);
load cottonclassificationnet45.mat;

%used my trained model 
net = cottonclassificationnet45;
net.Layers
analyzeNetwork(net)

inputSize = net.Layers(1).InputSize;

augmentedTrainingSet = augmentedImageDatastore(inputSize(1:2),imds);

%Feature extraction from last fully connected layer
layer = 'fc7';
bestcottonclassificationnet45featuresfc7 = activations(net,augmentedTrainingSet,layer,'OutputAs','rows');

%training lables
labelfc7 = imds.Labels;
save('labelfc7.mat','labelfc7');

%saving features
save('bestcottonclassificationnet45featuresfc7.mat','bestcottonclassificationnet45featuresfc7');

