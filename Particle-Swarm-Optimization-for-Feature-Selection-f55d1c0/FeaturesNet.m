clear all;
clc;
close all;

%uploading testing dataset for feature extraction purpose
imds = imageDatastore('G:\upcoming research\Cotton leaves Dataset for diseases classification using EfficienNet Model\Genetic Feature selection\Testing', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

numTrainImages = numel(imds.Labels);
load cottonclassificationnet.mat;

%used my trained model 
net = cottonclassificationnet;
net.Layers
analyzeNetwork(net)

inputSize = net.Layers(1).InputSize;

augmentedTrainingSet = augmentedImageDatastore(inputSize(1:2),imds);

%Feature extraction from last fully connected layer
layer = 'fc8';
bestcottonclassificationnetfeatures = activations(net,augmentedTrainingSet,layer,'OutputAs','rows');

%training lables
label = imds.Labels;
save('label.mat','label');

%saving features
save('bestcottonclassificationnetfeatures.mat','bestcottonclassificationnetfeatures');

