%% Binary Dragonfly Algorithm with Multiple Classifiers
clc; clear; close all;

% Total Execution Timer
totalExecutionStart = tic;

% Load dataset
load bestCCnetfeatureswithlables45fc8.mat;

% Set validation set percentage
ho = 0.2; 
% Hold-out method with no stratification
HO = cvpartition(labels, 'HoldOut', ho, 'Stratify', false);
 
% BDA parameter settings
N = 10; 
max_Iter = 100; 

% Capture initial memory use
user = memory();
initialMemory = user.MemUsedMATLAB / 1e6; % Memory use in MB

% Feature Selection Timer
featureSelectionStart = tic;

% Perform feature selection
[sFeat, Sf, Nf, curve] = jBDA(features, labels, N, max_Iter, HO);

% Feature Selection Timer Stop and Memory Calculation
featureSelectionTime = toc(featureSelectionStart);
user = memory();  % Retrieve memory info again after feature selection
finalMemory = user.MemUsedMATLAB / 1e6; % Convert bytes to MB
memoryUsed = finalMemory - initialMemory; % Calculate memory used

% Use only selected features
selectedFeatures = features(:, Sf);

% Classifier Training and Evaluation Timer Start
classifierPerformanceTimeStart = tic;

% KNN
modelKNN = fitcknn(selectedFeatures(HO.training,:), labels(HO.training));
predKNN = predict(modelKNN, selectedFeatures(HO.test,:));

% SVM with ECOC
modelSVM = fitcecoc(selectedFeatures(HO.training,:), labels(HO.training));
predSVM = predict(modelSVM, selectedFeatures(HO.test,:));

% Naive Bayes
modelNB = fitcnb(selectedFeatures(HO.training,:), labels(HO.training));
predNB = predict(modelNB, selectedFeatures(HO.test,:));

% Decision Tree
modelTree = fitctree(selectedFeatures(HO.training,:), labels(HO.training));
predTree = predict(modelTree, selectedFeatures(HO.test,:));

% Classifier Training and Evaluation Timer Stop
classifierPerformanceTime = toc(classifierPerformanceTimeStart);

% Calculate confusion matrices
CKNN = confusionmat(labels(HO.test), predKNN);
CSVM = confusionmat(labels(HO.test), predSVM);
CNB = confusionmat(labels(HO.test), predNB);
CTree = confusionmat(labels(HO.test), predTree);

% Display confusion matrices
disp('Confusion Matrix KNN:');
disp(CKNN);
disp('Confusion Matrix SVM:');
disp(CSVM);
disp('Confusion Matrix Naive Bayes:');
disp(CNB);
disp('Confusion Matrix Decision Tree:');
disp(CTree);

% Plot convergence curve
figure;
plot(1:max_Iter, curve);
xlabel('Number of Iterations');
ylabel('Fitness Value'); 
title('BDA Convergence Curve'); 
grid on;

% Save models, optional
save('BDA_Models.mat', 'modelKNN', 'modelSVM', 'modelNB', 'modelTree');

% Total Execution Timer
totalExecutionTime = toc(totalExecutionStart);

% Display execution times and memory usage
fprintf('Feature selection time: %.4f seconds.\n', featureSelectionTime);
fprintf('Memory used for feature selection: %.4f MB.\n', memoryUsed);
fprintf('Time taken for classifier training and testing: %.4f seconds.\n', classifierPerformanceTime);
fprintf('Total execution time: %.4f seconds (includes all operations).\n', totalExecutionTime);



% %% Binary Dragonfly Algorithm with Multiple Classifiers
% clc; clear; close all;
% 
% % Total Execution Timer
% totalExecutionStart = tic;
% 
% % Load dataset
% load bestCCnetfeatureswithlables45fc8.mat;
% 
% % Set validation set percentage
% ho = 0.2; 
% % Hold-out method with no stratification
% HO = cvpartition(labels, 'HoldOut', ho, 'Stratify', false);
% 
% % BDA parameter settings
% N = 10; 
% max_Iter = 100; 
% 
% % Capture initial memory use
% user = memory();
% initialMemory = user.MemUsedMATLAB / 1e6; % Memory use in MB
% 
% % Feature Selection Timer
% featureSelectionStart = tic;
% 
% % Perform feature selection
% [sFeat, Sf, Nf, curve] = jBDA(features, labels, N, max_Iter, HO);
% 
% % Feature Selection Timer Stop and Memory Calculation
% featureSelectionTime = toc(featureSelectionStart);
% user = memory();  % Retrieve memory info again after feature selection
% finalMemory = user.MemUsedMATLAB / 1e6; % Convert bytes to MB
% memoryUsed = finalMemory - initialMemory; % Calculate memory used
% 
% % Use only selected features
% selectedFeatures = features(:, Sf);
% 
% % Classifier Training and Evaluation Timer Start
% classifierPerformanceTimeStart = tic;
% 
% % Define classifiers
% classifiers = {'KNN', 'SVM', 'Naive Bayes', 'Decision Tree'};
% models = cell(1,4);
% predictions = cell(1,4);
% numFeaturesUsed = zeros(1,4);
% classificationResults = zeros(1,4); % Store classification accuracy for each model
% 
% % KNN - K-Nearest Neighbors
% modelKNN = fitcknn(selectedFeatures(HO.training,:), labels(HO.training));
% predKNN = predict(modelKNN, selectedFeatures(HO.test,:));
% numFeaturesUsed(1) = size(selectedFeatures, 2); % Number of features used by KNN is the size of the feature set
% models{1} = modelKNN;
% predictions{1} = predKNN;
% classificationResults(1) = sum(predKNN == labels(HO.test)) / length(labels(HO.test));
% 
% % SVM with ECOC
% modelSVM = fitcecoc(selectedFeatures(HO.training,:), labels(HO.training));
% predSVM = predict(modelSVM, selectedFeatures(HO.test,:));
% numFeaturesUsed(2) = size(selectedFeatures, 2); % Number of features used by SVM
% models{2} = modelSVM;
% predictions{2} = predSVM;
% classificationResults(2) = sum(predSVM == labels(HO.test)) / length(labels(HO.test));
% 
% % Naive Bayes
% modelNB = fitcnb(selectedFeatures(HO.training,:), labels(HO.training));
% predNB = predict(modelNB, selectedFeatures(HO.test,:));
% numFeaturesUsed(3) = size(selectedFeatures, 2); % Number of features used by Naive Bayes
% models{3} = modelNB;
% predictions{3} = predNB;
% classificationResults(3) = sum(predNB == labels(HO.test)) / length(labels(HO.test));
% 
% % Decision Tree
% modelTree = fitctree(selectedFeatures(HO.training,:), labels(HO.training));
% predTree = predict(modelTree, selectedFeatures(HO.test,:));
% numFeaturesUsed(4) = size(selectedFeatures, 2); % Number of features used by Decision Tree
% models{4} = modelTree;
% predictions{4} = predTree;
% classificationResults(4) = sum(predTree == labels(HO.test)) / length(labels(HO.test));
% 
% % Classifier Training and Evaluation Timer Stop
% classifierPerformanceTime = toc(classifierPerformanceTimeStart);
% 
% % Calculate confusion matrices
% CKNN = confusionmat(labels(HO.test), predKNN);
% CSVM = confusionmat(labels(HO.test), predSVM);
% CNB = confusionmat(labels(HO.test), predNB);
% CTree = confusionmat(labels(HO.test), predTree);
% 
% % Display confusion matrices
% disp('Confusion Matrix KNN:');
% disp(CKNN);
% disp('Confusion Matrix SVM:');
% disp(CSVM);
% disp('Confusion Matrix Naive Bayes:');
% disp(CNB);
% disp('Confusion Matrix Decision Tree:');
% disp(CTree);
% 
% % Display number of features used for each classifier
% disp('Number of features used by each classifier:');
% disp(['KNN: ', num2str(numFeaturesUsed(1))]);
% disp(['SVM: ', num2str(numFeaturesUsed(2))]);
% disp(['Naive Bayes: ', num2str(numFeaturesUsed(3))]);
% disp(['Decision Tree: ', num2str(numFeaturesUsed(4))]);
% 
% % Plot classification performance vs. classifier names
% figure;
% bar(classificationResults); % Using classification results (accuracy) as y-values
% set(gca, 'XTickLabel', classifiers); % X-axis as classifiers
% xlabel('Classifier');
% ylabel('Accuracy');
% title('Classification Performance of Different Classifiers');
% grid on;
% 
% % Plot convergence curve
% figure;
% plot(1:max_Iter, curve);
% xlabel('Number of Iterations');
% ylabel('Fitness Value'); 
% title('BDA Convergence Curve'); 
% grid on;
% 
% % Save models, optional
% save('BDA_Models.mat', 'models{1}', 'models{2}', 'models{3}', 'models{4}');
% 
% % Total Execution Timer
% totalExecutionTime = toc(totalExecutionStart);
% 
% % Display execution times and memory usage
% fprintf('Feature selection time: %.4f seconds.\n', featureSelectionTime);
% fprintf('Memory used for feature selection: %.4f MB.\n', memoryUsed);
% fprintf('Time taken for classifier training and testing: %.4f seconds.\n', classifierPerformanceTime);
% fprintf('Total execution time: %.4f seconds (includes all operations).\n', totalExecutionTime);
